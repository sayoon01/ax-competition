"""
개선 v3 — 올바른 방향으로 수정
==============================================
수정1: Train 2016-2022 → 임계값 튜닝(Val 2023) → 최종 모델 2016-2023 재학습
       새 모델의 확률 분포에 맞춰 임계값 재탐색 (0.01 ~ 0.10 세밀하게)
수정2: 14일 선행 모델 — 외인성 피처 유지(조류 제외) + Train 2016-2023으로 확장

실행: python3.10 improve_v3.py
결과: outputs/improvements_v3/
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
)
import lightgbm as lgb
import shap
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

def _set_font():
    for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p):
            fe = fm.FontEntry(fname=p, name="KorFont")
            fm.fontManager.ttflist.append(fe)
            plt.rcParams["font.family"] = "KorFont"
            return
_set_font()
plt.rcParams["axes.unicode_minus"] = False

BASE      = Path(__file__).parent
DATA_PATH = BASE / "finaldata.csv"
OUT       = BASE.parent / "outputs" / "improvements_v3"
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

STAGE_MAP  = {"미발령": 0, "관심": 1, "경계": 2, "조류대발생": 3}
LABELS     = ["미발령", "관심", "경계"]
CYANO_COLS = ["microcystis", "anabaena", "oscillatoria", "aphanizomenon"]

# 외인성 피처 키워드 (조류·Chl-a·이전 단계 제외)
EXOG_INCL = ["기온", "강수", "일사", "풍속", "습도", "전운",
             "수위", "저수", "유입", "방류", "gdd", "month",
             "sin_", "cos_", "수온", "pH", "DO", "탁도"]
EXOG_EXCL = ["total_cyano", "microcystis", "anabaena",
             "oscillatoria", "aphanizomenon", "Chl-a", "stage_lag",
             "stage_roll"]

# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def load_and_prep():
    df = pd.read_csv(DATA_PATH, parse_dates=["조사일"])
    df = df.sort_values(["채수위치", "조사일"]).reset_index(drop=True)
    df["stage_num"] = df["발령단계"].map(STAGE_MAP).fillna(0).astype(int)
    df = df.drop(columns=["일조시간 합계(hr)", "투명도"], errors="ignore")
    for site, g in df.groupby("채수위치"):
        idx = g.index
        df.loc[idx, CYANO_COLS] = g[CYANO_COLS].interpolate(method="linear", limit=14)
    df["total_cyano"] = df[CYANO_COLS].sum(axis=1)
    df["일강수량(mm)"] = df["일강수량(mm)"].fillna(df.get("강우량(mm)", 0))
    return df


def make_features(g: pd.DataFrame, lead: int = 7) -> pd.DataFrame:
    g = g.copy().sort_values("조사일")
    bio_cols = ["total_cyano"] + CYANO_COLS + ["Chl-a (㎎/㎥)"]
    for col in bio_cols:
        if col not in g.columns:
            continue
        for lag in [1, 3, 7, 14]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        for win in [3, 7, 14]:
            g[f"{col}_roll{win}m"]   = g[col].shift(1).rolling(win).mean()
            g[f"{col}_roll{win}max"] = g[col].shift(1).rolling(win).max()
    for col in ["수온(℃)", "pH", "DO(㎎/L)", "탁도"]:
        if col not in g.columns:
            continue
        for lag in [1, 3, 7]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()
    for col in ["평균기온(°C)", "최고기온(°C)", "합계 일사량(MJ/m2)",
                "평균 풍속(m/s)", "평균 상대습도(%)"]:
        if col not in g.columns:
            continue
        for lag in [1, 3, 7]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()
    for col in ["일강수량(mm)", "강우량(mm)"]:
        if col not in g.columns:
            continue
        for lag in [1, 3, 7, 14]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7sum"]  = g[col].shift(1).rolling(7).sum()
        g[f"{col}_roll14sum"] = g[col].shift(1).rolling(14).sum()
    for col in ["수위(EL.m)", "저수율(%)", "유입량(㎥/s)", "총방류량(㎥/s)"]:
        if col not in g.columns:
            continue
        for lag in [1, 3, 7]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()
    if "최고기온(°C)" in g.columns:
        g["gdd7"]  = g["최고기온(°C)"].shift(1).clip(lower=10).rolling(7).sum()
        g["gdd14"] = g["최고기온(°C)"].shift(1).clip(lower=10).rolling(14).sum()
    if "수온(℃)" in g.columns and "평균기온(°C)" in g.columns:
        g["temp_diff"]        = g["수온(℃)"] - g["평균기온(°C)"]
        g["temp_diff_lag1"]   = g["temp_diff"].shift(1)
        g["temp_diff_roll7m"] = g["temp_diff"].shift(1).rolling(7).mean()
    for lag in [1, 3, 7]:
        g[f"stage_lag{lag}"] = g["stage_num"].shift(lag)
    g["stage_roll7max"] = g["stage_num"].shift(1).rolling(7).max()
    g["month"]          = g["조사일"].dt.month
    g["week_of_year"]   = g["조사일"].dt.isocalendar().week.astype(int)
    g["sin_month"]      = np.sin(2 * np.pi * g["month"] / 12)
    g["cos_month"]      = np.cos(2 * np.pi * g["month"] / 12)
    g[f"target_d{lead}"] = g["stage_num"].shift(-lead)
    return g


EXCLUDE_KEYS = {"조사일", "채수위치", "발령단계", "stage_num", "일강수량(mm)"}

def get_all_feat(df):
    return [c for c in df.columns
            if c not in EXCLUDE_KEYS and df[c].dtype != object
            and not c.startswith("target_")]

def get_exog_feat(feat_cols):
    return [c for c in feat_cols
            if any(kw in c for kw in EXOG_INCL)
            and not any(ex in c for ex in EXOG_EXCL)]

def sw(y, boost=3.0):
    cnt = pd.Series(y).value_counts().sort_index()
    base = {k: len(y) / (len(cnt) * v) for k, v in cnt.items()}
    base[2] = base.get(2, 1.0) * boost
    return pd.Series(y).map(base).values

def apply_thr(prob, tb, tc):
    yp = np.zeros(len(prob), dtype=int)
    yp[prob[:, 2] >= tb] = 2
    yp[(prob[:, 1] >= tc) & (yp < 2)] = 1
    return yp

def mdict(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    yb_t = (yt >= 1).astype(int); yb_p = (yp >= 1).astype(int)
    b_m = yt == 2
    b_r = float(recall_score(yt[b_m], yp[b_m], labels=[2],
                             average="micro", zero_division=0)) if b_m.sum() > 0 else None
    return dict(
        accuracy       = round(float(accuracy_score(yt, yp)), 4),
        macro_f1       = round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        alert_recall   = round(float(recall_score(yb_t, yb_p, zero_division=0)), 4),
        alert_precision= round(float(precision_score(yb_t, yb_p, zero_division=0)), 4),
        alert_f1       = round(float(f1_score(yb_t, yb_p, zero_division=0)), 4),
        boundary_recall= round(b_r, 4) if b_r is not None else None,
    )

def cost_val(yt, yp):
    yb_t = (np.array(yt) >= 1).astype(int)
    yb_p = (np.array(yp) >= 1).astype(int)
    FP = int(((yb_p == 1) & (yb_t == 0)).sum())
    FN = int(((yb_p == 0) & (yb_t == 1)).sum())
    return FP + 3 * FN

def best_threshold(val_prob, y_val, tb_range, tc_range, min_b_rec=0.70):
    """cost 최소 + boundary_recall ≥ min_b_rec 제약 하 최적 임계값 탐색"""
    rows = []
    for tb in tb_range:
        for tc in tc_range:
            yp = apply_thr(val_prob, tb, tc)
            c  = cost_val(y_val, yp)
            b_m = np.array(y_val) == 2
            b_r = float((yp[b_m] == 2).sum()) / max(b_m.sum(), 1)
            rows.append({"tb": tb, "tc": tc, "cost": c, "b_rec": round(b_r, 3)})
    rows.sort(key=lambda x: x["cost"])
    # Recall 제약 만족하는 최소 비용
    filtered = [r for r in rows if r["b_rec"] >= min_b_rec]
    best = filtered[0] if filtered else rows[0]
    return best["tb"], best["tc"], best

def cm_heatmap(yt, yp, title, ax):
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    sns.heatmap(pd.DataFrame(cm, index=LABELS, columns=LABELS),
                annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("실제"); ax.set_xlabel("예측")

def timeseries_plot(sub, target_col, title, path):
    stage_colors = {0: "green", 1: "orange", 2: "red"}
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(sub["조사일"], sub["total_cyano"], color="#2196F3", lw=1.2, label="total_cyano")
    axes[0].axhline(1000,  color="orange", ls="--", lw=1, label="관심 기준(1,000)")
    axes[0].axhline(10000, color="red",    ls="--", lw=1, label="경계 기준(10,000)")
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_ylabel("유해남조류 (cells/mL)"); axes[0].legend(fontsize=8)
    axes[0].set_title(title, fontsize=11)
    for _, row in sub.iterrows():
        axes[1].axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
                        color=stage_colors.get(int(row[target_col]), "gray"), alpha=0.45)
        axes[1].axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["pred_stage"]), "gray"), alpha=0.15)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color="green", alpha=0.5, label="미발령"),
                             Patch(color="orange", alpha=0.5, label="관심"),
                             Patch(color="red",    alpha=0.5, label="경계"),
                             Patch(color="gray",   alpha=0.2, label="예측(연함)")],
                   loc="upper right", fontsize=8)
    axes[1].set_yticks([]); axes[1].set_title("발령단계: 실제(진함) vs 예측(연함)", fontsize=11)
    axes[2].stackplot(sub["조사일"],
                      sub["prob_normal"], sub["prob_caution"], sub["prob_alert"],
                      labels=["미발령", "관심", "경계"],
                      colors=["#66BB6A", "#FFA726", "#EF5350"], alpha=0.85)
    axes[2].set_ylabel("예측 확률"); axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right", fontsize=8); axes[2].set_title("경보 발령 확률", fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 공통 데이터 준비
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("데이터 준비")
print("=" * 70)

df_raw = load_and_prep()

# 7일 피처
parts7 = [make_features(g, lead=7) for _, g in df_raw.groupby("채수위치")]
df7    = pd.concat(parts7).sort_values(["조사일", "채수위치"]).reset_index(drop=True)
df7    = df7.dropna(subset=["target_d7"])
df7["target_d7"] = df7["target_d7"].astype(int).clip(upper=2)

ALL_FEAT = get_all_feat(df7)
print(f"  전체 피처: {len(ALL_FEAT)}개")

# 분리
tune_train = df7[df7["조사일"] < "2023-01-01"]   # 임계값 튜닝용 train
val_df     = df7[(df7["조사일"] >= "2023-01-01") & (df7["조사일"] < "2024-01-01")]  # val(2023)
full_train = df7[df7["조사일"] < "2024-01-01"]    # 최종 train (2016-2023)
test_df    = df7[df7["조사일"] >= "2024-01-01"]   # test (2024-2025)

for name, sub in [("튜닝Train(2016-2022)", tune_train), ("Val(2023)", val_df),
                   ("최종Train(2016-2023)", full_train), ("Test(2024-2025)", test_df)]:
    n = len(sub); bnd = (sub["target_d7"] == 2).sum()
    print(f"  {name}: {n:,}행 | 경계 {bnd}건({bnd/n*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 수정1: 7일 모델 — 2단계 학습 + 새 모델용 임계값 세밀 재탐색
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("수정1: 7일 선행 모델 — Train 확장 + 임계값 재탐색")
print("=" * 70)

Xtr = tune_train[ALL_FEAT]; ytr = tune_train["target_d7"]
Xvl = val_df[ALL_FEAT];     yvl = val_df["target_d7"]
Xfl = full_train[ALL_FEAT]; yfl = full_train["target_d7"]
Xte = test_df[ALL_FEAT];    yte = test_df["target_d7"]

# ── step A: 임계값 탐색 모델 (Train 2016-2022, Val 2023)
print("\n  [A] 임계값 탐색 모델 (Train 2016-2022)")
ros_a = RandomOverSampler(random_state=42, sampling_strategy={2: 1000})
Xtr_a, ytr_a = ros_a.fit_resample(Xtr, ytr)

lgb_params = dict(
    objective="multiclass", num_class=3, metric="multi_logloss",
    n_estimators=1500, learning_rate=0.03, max_depth=7, num_leaves=127,
    min_child_samples=10, feature_fraction=0.75, bagging_fraction=0.75,
    bagging_freq=5, lambda_l1=0.05, lambda_l2=0.1,
    verbose=-1, random_state=42, n_jobs=-1,
)
model_a = lgb.LGBMClassifier(**lgb_params)
model_a.fit(Xtr_a, ytr_a, sample_weight=sw(ytr_a), eval_set=[(Xvl, yvl)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
print(f"    best_iter={model_a.best_iteration_}")

vp_a = model_a.predict_proba(Xvl)

# 세밀 탐색: 0.01 ~ 0.20 (0.005 간격)
TB_RANGE = np.arange(0.01, 0.21, 0.005)
TC_RANGE = np.arange(0.10, 0.55, 0.05)
tb_a, tc_a, best_a = best_threshold(vp_a, yvl.values, TB_RANGE, TC_RANGE)
print(f"    최적 임계값: 경계={tb_a:.3f}, 관심={tc_a:.2f} "
      f"(val cost={best_a['cost']}, val b_rec={best_a['b_rec']})")

# ── step B: 최종 모델 (Train 2016-2023, 임계값은 A에서 구한 값 사용)
print("\n  [B] 최종 모델 (Train 2016-2023)")
n_b = (yfl == 2).sum()
ros_b = RandomOverSampler(random_state=42, sampling_strategy={2: min(n_b * 4, 1500)})
Xfl_sm, yfl_sm = ros_b.fit_resample(Xfl, yfl)
dist = pd.Series(yfl_sm).value_counts().sort_index()
print(f"    오버샘플 분포: { {int(k): int(v) for k, v in dist.items()} }")

lgb_params_b = {**lgb_params, "n_estimators": 2000}
model_b = lgb.LGBMClassifier(**lgb_params_b)
model_b.fit(Xfl_sm, yfl_sm, sample_weight=sw(yfl_sm), eval_set=[(Xvl, yvl)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
print(f"    best_iter={model_b.best_iteration_}")

tp_b = model_b.predict_proba(Xte)

# 새 모델의 확률 분포 확인 → 임계값 재탐색
print("\n  [C] 새 모델(B) 확률 분포 확인 & 임계값 재탐색")
vp_b = model_b.predict_proba(Xvl)

# 두 모델의 경계 클래스 확률 분포 비교
print(f"    모델A (2016-2022) 경계 확률 통계:")
print(f"      전체  — mean={vp_a[:,2].mean():.4f} med={np.median(vp_a[:,2]):.4f} max={vp_a[:,2].max():.4f}")
b_mask = yvl.values == 2
print(f"      경계  — mean={vp_a[b_mask,2].mean():.4f} med={np.median(vp_a[b_mask,2]):.4f}")
print(f"    모델B (2016-2023) 경계 확률 통계:")
print(f"      전체  — mean={vp_b[:,2].mean():.4f} med={np.median(vp_b[:,2]):.4f} max={vp_b[:,2].max():.4f}")
print(f"      경계  — mean={vp_b[b_mask,2].mean():.4f} med={np.median(vp_b[b_mask,2]):.4f}")

# 모델B 전용 임계값 재탐색 (더 촘촘하게)
TB_FINE = np.arange(0.005, 0.20, 0.003)
tb_b, tc_b, best_b = best_threshold(vp_b, yvl.values, TB_FINE, TC_RANGE)
print(f"\n    모델B 최적 임계값: 경계={tb_b:.3f}, 관심={tc_b:.2f} "
      f"(val cost={best_b['cost']}, val b_rec={best_b['b_rec']})")

yp_b = apply_thr(tp_b, tb_b, tc_b)
m_b  = mdict(yte.values, yp_b)
print(f"\n    최종 모델 테스트 성능: {m_b}")
print(f"\n  분류 리포트 (Test 2024-2025):")
print(classification_report(yte, yp_b, target_names=LABELS, zero_division=0))

# 연도별 성능
print("  연도별 성능:")
yearly7 = []
for yr in sorted(test_df["조사일"].dt.year.unique()):
    m_yr = test_df["조사일"].dt.year == yr
    m_ = mdict(yte[m_yr].values, yp_b[m_yr.values])
    yearly7.append({"year": yr, "n": int(m_yr.sum()), **m_})
    print(f"    {yr}: n={m_yr.sum():,}  acc={m_['accuracy']}  macro_f1={m_['macro_f1']}  boundary_recall={m_['boundary_recall']}")

# 전체 + 지점별 혼동행렬
fig, axes = plt.subplots(1, 4, figsize=(22, 4))
cm_heatmap(yte.values, yp_b,
           f"전체\nboundary_recall={m_b['boundary_recall']}", axes[0])
for ax, site in zip(axes[1:], test_df["채수위치"].unique()):
    s_m = test_df["채수위치"].values == site
    m_s = mdict(yte.values[s_m], yp_b[s_m])
    cm_heatmap(yte.values[s_m], yp_b[s_m],
               f"{site}\nboundary_recall={m_s['boundary_recall']}", ax)
plt.suptitle("수정1 최종 모델: 7일 선행 예측 (Train 2016-2023, Test 2024-2025)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "v3_7d_confusion.png", dpi=150); plt.close()

# 지점별 시계열 플롯
print("  지점별 시계열 플롯 저장...")
tp_b_all = test_df.copy()
tp_b_all["pred_stage"]   = yp_b
tp_b_all["prob_normal"]  = tp_b[:, 0]
tp_b_all["prob_caution"] = tp_b[:, 1]
tp_b_all["prob_alert"]   = tp_b[:, 2]
tp_b_all.to_csv(OUT / "reports" / "predictions_7d_v3.csv", index=False, encoding="utf-8-sig")

for site in test_df["채수위치"].unique():
    sub = tp_b_all[tp_b_all["채수위치"] == site].sort_values("조사일")
    timeseries_plot(sub, "target_d7",
                    f"{site} — 유해남조류 농도 (Test 2024-2025)",
                    OUT / "plots" / f"v3_7d_timeseries_{site}.png")
    print(f"    저장: v3_7d_timeseries_{site}.png")

# SHAP
print("  SHAP 분석...")
expl_b = shap.TreeExplainer(model_b)
samp   = Xte.sample(min(400, len(Xte)), random_state=42)
sv     = expl_b.shap_values(samp)
if isinstance(sv, np.ndarray) and sv.ndim == 3:
    sv = [sv[:, :, c] for c in range(sv.shape[2])]
mean_sv = sum(np.abs(s).mean(axis=0) for s in sv)
shap7 = pd.DataFrame({"feature": ALL_FEAT, "mean_abs_shap": mean_sv})
shap7 = shap7.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

top20 = shap7.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="#1565C0")
ax.set_xlabel("평균 |SHAP|")
ax.set_title("v3 7일 모델 피처 중요도 Top-20 (Train 2016-2023)", fontsize=12)
plt.tight_layout(); fig.savefig(OUT / "plots" / "v3_7d_shap.png", dpi=150); plt.close()
print(f"  SHAP Top-5: {shap7['feature'].head(5).tolist()}")

result7 = {"train": "2016-2023", "test": "2024-2025",
           "thr_boundary": float(tb_b), "thr_caution": float(tc_b),
           "metrics": m_b, "shap_top5": shap7["feature"].head(5).tolist()}

# ═══════════════════════════════════════════════════════════════════════════════
# 수정2: 14일 선행 모델 — 외인성 피처 유지 + Train 2016-2023 확장
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("수정2: 14일 선행 모델 — 외인성 피처 유지 + Train 2016-2023")
print("=" * 70)

parts14 = [make_features(g, lead=14) for _, g in df_raw.groupby("채수위치")]
df14    = pd.concat(parts14).sort_values(["조사일", "채수위치"]).reset_index(drop=True)
df14    = df14.dropna(subset=["target_d14"])
df14["target_d14"] = df14["target_d14"].astype(int).clip(upper=2)

ALL_FEAT14  = get_all_feat(df14)
EXOG_FEAT14 = get_exog_feat(ALL_FEAT14)
print(f"  전체 피처: {len(ALL_FEAT14)}개 → 외인성: {len(EXOG_FEAT14)}개 (조류 제외)")

# 분리
tune_tr14 = df14[df14["조사일"] < "2023-01-01"]
val_14    = df14[(df14["조사일"] >= "2023-01-01") & (df14["조사일"] < "2024-01-01")]
full_14   = df14[df14["조사일"] < "2024-01-01"]   # 2016-2023 전체
test_14   = df14[df14["조사일"] >= "2024-01-01"]

for name, sub in [("튜닝Train14(2016-2022)", tune_tr14), ("Val14(2023)", val_14),
                   ("최종Train14(2016-2023)", full_14),   ("Test14(2024-2025)", test_14)]:
    n = len(sub); bnd = (sub["target_d14"] == 2).sum()
    print(f"  {name}: {n:,}행 | 경계 {bnd}건({bnd/n*100:.1f}%)")

Xtr14 = tune_tr14[EXOG_FEAT14]; ytr14 = tune_tr14["target_d14"]
Xvl14 = val_14[EXOG_FEAT14];    yvl14 = val_14["target_d14"]
Xfl14 = full_14[EXOG_FEAT14];   yfl14 = full_14["target_d14"]
Xte14 = test_14[EXOG_FEAT14];   yte14 = test_14["target_d14"]

# ── step A: 임계값 탐색 (Train 2016-2022, Val 2023)
print("\n  [A] 14일 임계값 탐색 모델 (Train 2016-2022)")
n_b14 = (ytr14 == 2).sum()
ros_14a = RandomOverSampler(random_state=42, sampling_strategy={2: min(n_b14 * 4, 1000)})
Xtr14_sm, ytr14_sm = ros_14a.fit_resample(Xtr14, ytr14)

lgb14_params = {**lgb_params, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 63}
model_14a = lgb.LGBMClassifier(**lgb14_params)
model_14a.fit(Xtr14_sm, ytr14_sm, sample_weight=sw(ytr14_sm), eval_set=[(Xvl14, yvl14)],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
print(f"    best_iter={model_14a.best_iteration_}")

vp_14a = model_14a.predict_proba(Xvl14)
tb_14a, tc_14a, best_14a = best_threshold(vp_14a, yvl14.values, TB_RANGE, TC_RANGE)
print(f"    임계값 (2016-2022 기반): 경계={tb_14a:.3f}, 관심={tc_14a:.2f} "
      f"(val cost={best_14a['cost']}, b_rec={best_14a['b_rec']})")

# ── step B: 최종 14일 모델 (Train 2016-2023)
print("\n  [B] 14일 최종 모델 (Train 2016-2023)")
n_bfl14 = (yfl14 == 2).sum()
ros_14b = RandomOverSampler(random_state=42, sampling_strategy={2: min(n_bfl14 * 4, 1500)})
Xfl14_sm, yfl14_sm = ros_14b.fit_resample(Xfl14, yfl14)

lgb14f_params = {**lgb14_params, "n_estimators": 1500, "learning_rate": 0.04, "max_depth": 7, "num_leaves": 127}
model_14b = lgb.LGBMClassifier(**lgb14f_params)
model_14b.fit(Xfl14_sm, yfl14_sm, sample_weight=sw(yfl14_sm), eval_set=[(Xvl14, yvl14)],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
print(f"    best_iter={model_14b.best_iteration_}")

tp_14b  = model_14b.predict_proba(Xte14)
vp_14b  = model_14b.predict_proba(Xvl14)

# 새 모델용 임계값 재탐색
tb_14b, tc_14b, best_14b = best_threshold(vp_14b, yvl14.values, TB_FINE, TC_RANGE)
print(f"    모델B 임계값: 경계={tb_14b:.3f}, 관심={tc_14b:.2f} "
      f"(val cost={best_14b['cost']}, b_rec={best_14b['b_rec']})")

yp_14b = apply_thr(tp_14b, tb_14b, tc_14b)
m_14b  = mdict(yte14.values, yp_14b)
print(f"\n    14일 최종 모델 테스트 성능: {m_14b}")
print(f"\n  분류 리포트 (14일, Test 2024-2025):")
print(classification_report(yte14, yp_14b, target_names=LABELS, zero_division=0))

# 혼동행렬 (7일 vs 14일)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_heatmap(yte.values, yp_b,
           f"7일 선행\n(boundary_recall={m_b['boundary_recall']})", axes[0])
cm_heatmap(yte14.values, yp_14b,
           f"14일 선행 (외인성)\n(boundary_recall={m_14b['boundary_recall']})", axes[1])
plt.suptitle("7일 vs 14일 선행 예측 비교 (Test 2024-2025)", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "v3_7d_vs_14d_confusion.png", dpi=150); plt.close()

# 지점별 시계열 (14일)
print("  14일 지점별 시계열 플롯 저장...")
tp14_all = test_14.copy()
tp14_all["pred_stage"]   = yp_14b
tp14_all["prob_normal"]  = tp_14b[:, 0]
tp14_all["prob_caution"] = tp_14b[:, 1]
tp14_all["prob_alert"]   = tp_14b[:, 2]
tp14_all.to_csv(OUT / "reports" / "predictions_14d_v3.csv", index=False, encoding="utf-8-sig")

for site in test_14["채수위치"].unique():
    sub = tp14_all[tp14_all["채수위치"] == site].sort_values("조사일")
    timeseries_plot(sub, "target_d14",
                    f"{site} — 유해남조류 농도 (14일 선행, Test 2024-2025)",
                    OUT / "plots" / f"v3_14d_timeseries_{site}.png")
    print(f"    저장: v3_14d_timeseries_{site}.png")

# SHAP (14일)
expl_14 = shap.TreeExplainer(model_14b)
samp14  = Xte14.sample(min(300, len(Xte14)), random_state=42)
sv14    = expl_14.shap_values(samp14)
if isinstance(sv14, np.ndarray) and sv14.ndim == 3:
    sv14 = [sv14[:, :, c] for c in range(sv14.shape[2])]
mean14  = sum(np.abs(s).mean(axis=0) for s in sv14)
shap14  = pd.DataFrame({"feature": EXOG_FEAT14, "mean_abs_shap": mean14})
shap14  = shap14.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

top20_14 = shap14.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_14["feature"][::-1], top20_14["mean_abs_shap"][::-1], color="#B71C1C")
ax.set_xlabel("평균 |SHAP|")
ax.set_title("v3 14일 모델 피처 중요도 Top-20 (외인성 피처, Train 2016-2023)", fontsize=12)
plt.tight_layout(); fig.savefig(OUT / "plots" / "v3_14d_shap.png", dpi=150); plt.close()
print(f"  14일 SHAP Top-5: {shap14['feature'].head(5).tolist()}")

result14 = {"train": "2016-2023", "test": "2024-2025",
            "n_exog_features": len(EXOG_FEAT14),
            "thr_boundary": float(tb_14b), "thr_caution": float(tc_14b),
            "metrics": m_14b, "shap_top5": shap14["feature"].head(5).tolist()}

# ═══════════════════════════════════════════════════════════════════════════════
# 최종 비교
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("최종 비교 요약 — 전체 개선 히스토리")
print("=" * 70)

rows = [
    ("① 기존 (Train~2022, 임계값 0.12)",       {"boundary_recall": 0.1473, "alert_f1": 0.9315, "macro_f1": 0.5860}),
    ("② v1 cost-optimal (임계값 0.05)",         {"boundary_recall": 0.5241, "alert_f1": 0.9282, "macro_f1": 0.6631}),
    ("③ v3 수정1: 7일 (Train~2023, 재탐색)",     m_b),
    ("④ v3 수정2: 14일 외인성 (Train~2023)",     m_14b),
]
print(f"  {'항목':<42} | {'boundary_recall':>15} | {'alert_f1':>9} | {'macro_f1':>9}")
print("  " + "-" * 82)
for label, m in rows:
    br = m.get("boundary_recall") or 0.0
    print(f"  {label:<42} | {br:>15.4f} | {m['alert_f1']:>9.4f} | {m['macro_f1']:>9.4f}")

summary = {
    "note": "임계값 탐색: Train 2016-2022 모델 → Val 2023. 최종 모델: Train 2016-2023. Test: 2024-2025.",
    "result_7day":  result7,
    "result_14day": result14,
}
with open(OUT / "reports" / "v3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"\n  모든 결과: {OUT}")
print("완료!")
