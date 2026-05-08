"""
개선 v2 — 두 가지 수정
=====================================================
수정 1: 전략적 Train/Val/Test 재분리
  - 기존: Train(2016-2022) / Val(2023) / Test(2024+)
  - 변경: 임계값 튜닝은 2023(경계 70건)으로 유지,
          최종 모델은 2016-2023 전체로 재학습 → Test(2024-2025)
  → Train 경계 케이스: 105건 → 175건 (+70건)

수정 2: Task3 재수행 — 14일 선행 모델에 조류 데이터 포함
  - 기존: 외인성(기상/댐) 피처만 79개
  - 변경: 전체 피처(조류 포함) 사용

실행: python3.10 improve_v2.py
결과: outputs/improvements_v2/
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
)
import lightgbm as lgb
import shap
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
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

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_PATH = BASE / "finaldata.csv"
OUT       = BASE.parent / "outputs" / "improvements_v2"
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

STAGE_MAP  = {"미발령": 0, "관심": 1, "경계": 2, "조류대발생": 3}
LABELS     = ["미발령", "관심", "경계"]
CYANO_COLS = ["microcystis", "anabaena", "oscillatoria", "aphanizomenon"]

# =============================================================================
# 공통 전처리 & 피처 엔지니어링
# =============================================================================
def load_and_prep(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["조사일"])
    df = df.sort_values(["채수위치", "조사일"]).reset_index(drop=True)
    df["stage_num"] = df["발령단계"].map(STAGE_MAP).fillna(0).astype(int)
    df = df.drop(columns=["일조시간 합계(hr)", "투명도"], errors="ignore")
    for site, g in df.groupby("채수위치"):
        idx = g.index
        df.loc[idx, CYANO_COLS] = g[CYANO_COLS].interpolate(method="linear", limit=14)
    df["total_cyano"] = df[CYANO_COLS].sum(axis=1)
    df["일강수량(mm)"] = df["일강수량(mm)"].fillna(df.get("강우량(mm)", 0))
    return df


def make_features(g: pd.DataFrame, lead_days: int = 7) -> pd.DataFrame:
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
    g[f"target_d{lead_days}"] = g["stage_num"].shift(-lead_days)
    return g


EXCLUDE = {"조사일", "채수위치", "발령단계", "stage_num",
           "target_d7", "target_d14", "일강수량(mm)"}

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in EXCLUDE and df[c].dtype != object
            and not c.startswith("target_")]


def lgb_params(n_est=1500):
    return dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        n_estimators=n_est, learning_rate=0.03, max_depth=7, num_leaves=127,
        min_child_samples=10, feature_fraction=0.75, bagging_fraction=0.75,
        bagging_freq=5, lambda_l1=0.05, lambda_l2=0.1,
        verbose=-1, random_state=42, n_jobs=-1,
    )


def sample_weights(y, boost_cls=2, boost=3.0):
    counts = pd.Series(y).value_counts().sort_index()
    base   = {k: len(y) / (len(counts) * v) for k, v in counts.items()}
    base[boost_cls] = base.get(boost_cls, 1.0) * boost
    return pd.Series(y).map(base).values


def apply_threshold(prob, thr_b, thr_c):
    yp = np.zeros(len(prob), dtype=int)
    yp[prob[:, 2] >= thr_b] = 2
    yp[(prob[:, 1] >= thr_c) & (yp < 2)] = 1
    return yp


def metrics_dict(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    yb_t = (yt >= 1).astype(int)
    yb_p = (yp >= 1).astype(int)
    b_mask = yt == 2
    b_rec  = float(recall_score(yt[b_mask], yp[b_mask], labels=[2],
                                average="micro", zero_division=0)) if b_mask.sum() > 0 else None
    return dict(
        accuracy=round(float(accuracy_score(yt, yp)), 4),
        macro_f1=round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        alert_recall=round(float(recall_score(yb_t, yb_p, zero_division=0)), 4),
        alert_precision=round(float(precision_score(yb_t, yb_p, zero_division=0)), 4),
        alert_f1=round(float(f1_score(yb_t, yb_p, zero_division=0)), 4),
        boundary_recall=round(b_rec, 4) if b_rec is not None else None,
    )


def cost_metric(yt, yp, w_fp=1.0, w_fn=3.0):
    yb_t = (np.array(yt) >= 1).astype(int)
    yb_p = (np.array(yp) >= 1).astype(int)
    FP = int(((yb_p == 1) & (yb_t == 0)).sum())
    FN = int(((yb_p == 0) & (yb_t == 1)).sum())
    return w_fp * FP + w_fn * FN


def confusion_heatmap(yt, yp, title, ax):
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("실제"); ax.set_xlabel("예측")


# =============================================================================
# 데이터 준비 (7일 타겟)
# =============================================================================
print("=" * 70)
print("데이터 로드 & 피처 엔지니어링")
print("=" * 70)

df_raw = load_and_prep()
parts = [make_features(g, lead_days=7) for _, g in df_raw.groupby("채수위치")]
df7   = pd.concat(parts).sort_values(["조사일", "채수위치"]).reset_index(drop=True)
df7   = df7.dropna(subset=["target_d7"])
df7["target_d7"] = df7["target_d7"].astype(int).clip(upper=2)

FEAT = get_feature_cols(df7)
print(f"  총 피처 수: {len(FEAT)}")

# 연도별 경계 분포 출력
df7["year"] = df7["조사일"].dt.year
yd = df7.groupby(["year", "target_d7"]).size().unstack(fill_value=0)
print(f"\n  연도별 7일 선행 타겟 분포 (경계=2):")
print(yd.to_string())

# =============================================================================
# 수정 1: 전략적 2단계 Train/Val/Test 분리
# =============================================================================
print("\n" + "=" * 70)
print("수정 1: 전략적 Train/Val/Test 분리")
print("=" * 70)

# ── 단계 A: 임계값 튜닝용 (기존 방식)
#    Train: 2016-2022 / Val: 2023 (경계 70건) → 최적 임계값 도출
# ── 단계 B: 최종 모델
#    Train: 2016-2023 (경계 175건) / Test: 2024-2025

tune_train = df7[df7["조사일"] < "2023-01-01"]
tune_val   = df7[(df7["조사일"] >= "2023-01-01") & (df7["조사일"] < "2024-01-01")]
test_df    = df7[df7["조사일"] >= "2024-01-01"]
full_train = df7[df7["조사일"] < "2024-01-01"]   # 2016-2023 통합

def split_summary(name, sub, tgt_col="target_d7"):
    n = len(sub)
    col = tgt_col if tgt_col in sub.columns else [c for c in sub.columns if c.startswith("target_")][0]
    bnd = (sub[col] == 2).sum()
    ctn = (sub[col] == 1).sum()
    print(f"  {name}: {n:,}행 | 경계 {bnd}건({bnd/n*100:.1f}%) | 관심 {ctn}건({ctn/n*100:.1f}%)")

split_summary("튜닝Train (2016-2022)", tune_train)
split_summary("튜닝Val   (2023     )", tune_val)
split_summary("최종Train (2016-2023)", full_train)
split_summary("Test      (2024-2025)", test_df)

# ── A: 튜닝 모델 → 임계값 최적화
print("\n  [A] 임계값 튜닝 모델 학습 (Train 2016-2022, Val 2023)")
Xtr_t, ytr_t = tune_train[FEAT], tune_train["target_d7"]
Xvl_t, yvl_t = tune_val[FEAT],   tune_val["target_d7"]

ros = RandomOverSampler(random_state=42, sampling_strategy={2: 1000})
Xtr_sm, ytr_sm = ros.fit_resample(Xtr_t, ytr_t)

tune_model = lgb.LGBMClassifier(**lgb_params())
tune_model.fit(
    Xtr_sm, ytr_sm,
    sample_weight=sample_weights(ytr_sm),
    eval_set=[(Xvl_t, yvl_t)],
    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
)
print(f"    best_iter={tune_model.best_iteration_}")

vp = tune_model.predict_proba(Xvl_t)

# cost-optimal 임계값 탐색 (Val 2023 기준)
best_cost, BEST_THR_B, BEST_THR_C = 1e9, 0.10, 0.30
cost_rows = []
for tb in np.arange(0.04, 0.45, 0.01):
    for tc in np.arange(0.10, 0.55, 0.05):
        yp_v = apply_threshold(vp, tb, tc)
        c    = cost_metric(yvl_t, yp_v)
        b_m  = yvl_t.values == 2
        b_r  = float((yp_v[b_m] == 2).sum()) / max(b_m.sum(), 1)
        cost_rows.append({"tb": tb, "tc": tc, "cost": c, "b_rec": round(b_r, 3)})
        if c < best_cost:
            best_cost, BEST_THR_B, BEST_THR_C = c, tb, tc

# Recall ≥ 0.70 제약 하 최소 비용
filtered = [r for r in cost_rows if r["b_rec"] >= 0.70]
if filtered:
    best_r = min(filtered, key=lambda x: x["cost"])
    BEST_THR_B, BEST_THR_C = best_r["tb"], best_r["tc"]
    print(f"    최적 임계값 (Recall≥70%): 경계={BEST_THR_B:.2f}, 관심={BEST_THR_C:.2f} "
          f"val-cost={best_r['cost']} val-b_rec={best_r['b_rec']}")
else:
    print(f"    최적 임계값 (cost): 경계={BEST_THR_B:.2f}, 관심={BEST_THR_C:.2f} val-cost={best_cost}")

# ── B: 최종 모델 학습 (2016-2023 전체)
print("\n  [B] 최종 모델 학습 (Train 2016-2023)")
Xtr_f, ytr_f = full_train[FEAT], full_train["target_d7"]
Xte_f, yte_f = test_df[FEAT],    test_df["target_d7"]

n_b = (ytr_f == 2).sum()
sm_target = min(n_b * 4, 1500)
ros2 = RandomOverSampler(random_state=42, sampling_strategy={2: sm_target})
Xtr_f_sm, ytr_f_sm = ros2.fit_resample(Xtr_f, ytr_f)

dist = pd.Series(ytr_f_sm).value_counts().sort_index()
print(f"    오버샘플 후 분포: { {int(k): int(v) for k, v in dist.items()} }")

final_model = lgb.LGBMClassifier(**lgb_params(n_est=1800))
# early stopping을 위한 val: tune_val(2023) 사용
final_model.fit(
    Xtr_f_sm, ytr_f_sm,
    sample_weight=sample_weights(ytr_f_sm),
    eval_set=[(Xvl_t, yvl_t)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
)
print(f"    best_iter={final_model.best_iteration_}")

test_prob = final_model.predict_proba(Xte_f)
yp_final  = apply_threshold(test_prob, BEST_THR_B, BEST_THR_C)
m_final   = metrics_dict(yte_f.values, yp_final)
print(f"\n    최종 모델 테스트 성능: {m_final}")

print("\n  분류 리포트 (Test 2024-2025):")
print(classification_report(yte_f, yp_final, target_names=LABELS, zero_division=0))

# ── 혼동행렬 — 전체 및 지점별
print("  혼동행렬 저장...")
fig, axes = plt.subplots(1, 4, figsize=(22, 4))

confusion_heatmap(yte_f.values, yp_final,
                  f"전체 (Test 2024-2025)\nboundary_recall={m_final['boundary_recall']}", axes[0])

for ax, site in zip(axes[1:], df7["채수위치"].unique()):
    site_mask = test_df["채수위치"] == site
    yt_s = yte_f[site_mask].values
    yp_s = yp_final[site_mask.values]
    m_s  = metrics_dict(yt_s, yp_s)
    confusion_heatmap(yt_s, yp_s,
                      f"{site}\nboundary_recall={m_s['boundary_recall']}", ax)

plt.suptitle("수정1: 전략적 분리 + 최종 모델 (Train 2016-2023, Test 2024-2025)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "v2_confusion_final.png", dpi=150)
plt.close()

# ── 지점별 시계열 예측 플롯 (7일 선행)
print("  지점별 시계열 예측 플롯 저장...")
test_plot = test_df.copy()
test_plot["pred_stage"]  = yp_final
test_plot["prob_normal"] = test_prob[:, 0]
test_plot["prob_caution"]= test_prob[:, 1]
test_plot["prob_alert"]  = test_prob[:, 2]
test_plot.to_csv(OUT / "reports" / "holdout_predictions_v2.csv",
                 index=False, encoding="utf-8-sig")

stage_colors = {0: "green", 1: "orange", 2: "red"}

for site in df7["채수위치"].unique():
    sub = test_plot[test_plot["채수위치"] == site].sort_values("조사일")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # 남조류 농도
    axes[0].plot(sub["조사일"], sub["total_cyano"],
                 color="#2196F3", lw=1.2, label="total_cyano")
    axes[0].axhline(1000,  color="orange", ls="--", lw=1, label="관심 기준")
    axes[0].axhline(10000, color="red",    ls="--", lw=1, label="경계 기준")
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_ylabel("유해남조류 (cells/mL)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{site} — 유해남조류 농도 (2024-2025 Test)", fontsize=11)

    # 실제 vs 예측 단계
    for _, row in sub.iterrows():
        axes[1].axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["target_d7"]), "gray"), alpha=0.45)
        axes[1].axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["pred_stage"]), "gray"), alpha=0.15)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color="green",  alpha=0.5, label="미발령"),
        Patch(color="orange", alpha=0.5, label="관심"),
        Patch(color="red",    alpha=0.5, label="경계"),
        Patch(color="gray",   alpha=0.2, label="예측(연함)"),
    ], loc="upper right", fontsize=8)
    axes[1].set_yticks([])
    axes[1].set_title("발령단계: 실제(진함) vs 예측(연함) — 7일 선행", fontsize=11)

    # 예측 확률
    axes[2].stackplot(sub["조사일"],
                      sub["prob_normal"], sub["prob_caution"], sub["prob_alert"],
                      labels=["미발령", "관심", "경계"],
                      colors=["#66BB6A", "#FFA726", "#EF5350"], alpha=0.85)
    axes[2].set_ylabel("예측 확률")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("7일 선행 경보 발령 확률", fontsize=11)

    plt.suptitle(f"대청댐 {site} — 7일 선행 남조류 경보 예측 (최종 모델, Train 2016-2023)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "plots" / f"v2_timeseries_{site}.png", dpi=150)
    plt.close()
    print(f"    저장: v2_timeseries_{site}.png")

# ── SHAP 중요도 (최종 모델)
print("  SHAP 분석...")
explainer = shap.TreeExplainer(final_model)
shap_sample = Xte_f.sample(min(400, len(Xte_f)), random_state=42)
sv = explainer.shap_values(shap_sample)
if isinstance(sv, np.ndarray) and sv.ndim == 3:
    sv = [sv[:, :, c] for c in range(sv.shape[2])]
mean_sv = sum(np.abs(s).mean(axis=0) for s in sv)
shap_df = pd.DataFrame({"feature": FEAT, "mean_abs_shap": mean_sv})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

top20 = shap_df.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="#1565C0")
ax.set_xlabel("평균 |SHAP| (전체 클래스 합산)")
ax.set_title("v2 최종 모델 — 피처 중요도 Top-20", fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "plots" / "v2_shap_top20.png", dpi=150)
plt.close()
print(f"  SHAP Top-5: {shap_df['feature'].head(5).tolist()}")

# 연도별 성능 (Test 2024-2025)
print("\n  연도별 성능 분석:")
yearly = []
test_plot2 = test_plot.copy()
for yr in sorted(test_plot2["조사일"].dt.year.unique()):
    mask = test_plot2["조사일"].dt.year == yr
    yt_ = test_plot2.loc[mask, "target_d7"].values
    yp_ = test_plot2.loc[mask, "pred_stage"].values
    m   = metrics_dict(yt_, yp_)
    yearly.append({"year": yr, "n": len(yt_), **m})
    print(f"    {yr}: n={len(yt_):,}  acc={m['accuracy']}  macro_f1={m['macro_f1']}  "
          f"boundary_recall={m['boundary_recall']}")

pd.DataFrame(yearly).to_csv(OUT / "reports" / "yearly_metrics_v2.csv",
                              index=False, encoding="utf-8-sig")

result_v2_split = {"metrics": m_final, "threshold_boundary": BEST_THR_B,
                   "threshold_caution": BEST_THR_C, "yearly": yearly}

# =============================================================================
# 수정 2: 14일 선행 모델 — 전체 피처(조류 포함) 사용
# =============================================================================
print("\n" + "=" * 70)
print("수정 2: 14일 선행 모델 — 전체 피처(조류 포함)")
print("=" * 70)

parts14 = [make_features(g, lead_days=14) for _, g in df_raw.groupby("채수위치")]
df14    = pd.concat(parts14).sort_values(["조사일", "채수위치"]).reset_index(drop=True)
df14    = df14.dropna(subset=["target_d14"])
df14["target_d14"] = df14["target_d14"].astype(int).clip(upper=2)

FEAT14 = get_feature_cols(df14)
print(f"  14일 모델 전체 피처 수: {len(FEAT14)}")

# 같은 분리 기준 적용
tune_train14 = df14[df14["조사일"] < "2023-01-01"]
tune_val14   = df14[(df14["조사일"] >= "2023-01-01") & (df14["조사일"] < "2024-01-01")]
full_train14 = df14[df14["조사일"] < "2024-01-01"]
test14       = df14[df14["조사일"] >= "2024-01-01"]

split_summary("14일 최종Train (2016-2023)", full_train14, "target_d14")
split_summary("14일 Test      (2024-2025)", test14,       "target_d14")

# 임계값은 7일 모델과 동일하게 사용 (BEST_THR_B, BEST_THR_C)
# 단, 14일 모델 전용 임계값도 별도 탐색

Xtr14_v  = tune_train14[FEAT14];  ytr14_v = tune_train14["target_d14"]
Xvl14_v  = tune_val14[FEAT14];    yvl14_v = tune_val14["target_d14"]
Xtr14_f  = full_train14[FEAT14];  ytr14_f = full_train14["target_d14"]
Xte14    = test14[FEAT14];        yte14   = test14["target_d14"]

# 임계값 탐색 (Val 2023 기준)
print("\n  [A] 14일 모델 임계값 탐색 (Val 2023)...")
n_b14 = (ytr14_v == 2).sum()
ros14 = RandomOverSampler(random_state=42,
                          sampling_strategy={2: min(n_b14 * 4, 1000)})
Xtr14_sm, ytr14_sm = ros14.fit_resample(Xtr14_v, ytr14_v)

params14 = lgb_params(n_est=1200)
params14.update(learning_rate=0.05, max_depth=6, num_leaves=63)
tune14 = lgb.LGBMClassifier(**params14)
tune14.fit(
    Xtr14_sm, ytr14_sm,
    sample_weight=sample_weights(ytr14_sm),
    eval_set=[(Xvl14_v, yvl14_v)],
    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
)
vp14 = tune14.predict_proba(Xvl14_v)

best_cost14, THR14_B, THR14_C = 1e9, 0.10, 0.30
cost14_rows = []
for tb in np.arange(0.04, 0.45, 0.01):
    for tc in np.arange(0.10, 0.55, 0.05):
        yp_v = apply_threshold(vp14, tb, tc)
        c    = cost_metric(yvl14_v, yp_v)
        b_m  = yvl14_v.values == 2
        b_r  = float((yp_v[b_m] == 2).sum()) / max(b_m.sum(), 1)
        cost14_rows.append({"tb": tb, "tc": tc, "cost": c, "b_rec": round(b_r, 3)})
        if c < best_cost14:
            best_cost14, THR14_B, THR14_C = c, tb, tc

filtered14 = [r for r in cost14_rows if r["b_rec"] >= 0.70]
if filtered14:
    best_r14 = min(filtered14, key=lambda x: x["cost"])
    THR14_B, THR14_C = best_r14["tb"], best_r14["tc"]
    print(f"    14일 최적 임계값 (Recall≥70%): 경계={THR14_B:.2f}, 관심={THR14_C:.2f} "
          f"val-b_rec={best_r14['b_rec']}")
else:
    print(f"    14일 최적 임계값 (cost): 경계={THR14_B:.2f}, 관심={THR14_C:.2f}")

# 최종 14일 모델 학습 (2016-2023)
print("\n  [B] 14일 최종 모델 학습 (Train 2016-2023)...")
n_b14f = (ytr14_f == 2).sum()
ros14f = RandomOverSampler(random_state=42,
                           sampling_strategy={2: min(n_b14f * 4, 1500)})
Xtr14_fsm, ytr14_fsm = ros14f.fit_resample(Xtr14_f, ytr14_f)

params14f = lgb_params(n_est=1500)
params14f.update(learning_rate=0.04, max_depth=7, num_leaves=127)
final14 = lgb.LGBMClassifier(**params14f)
final14.fit(
    Xtr14_fsm, ytr14_fsm,
    sample_weight=sample_weights(ytr14_fsm),
    eval_set=[(Xvl14_v, yvl14_v)],
    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
)
print(f"    best_iter={final14.best_iteration_}")

tp14     = final14.predict_proba(Xte14)
yp14     = apply_threshold(tp14, THR14_B, THR14_C)
m14      = metrics_dict(yte14.values, yp14)
print(f"\n    14일 전체피처 최종 모델: {m14}")
print("\n  분류 리포트 (14일, Test 2024-2025):")
print(classification_report(yte14, yp14, target_names=LABELS, zero_division=0))

# SHAP (14일)
print("  14일 SHAP 분석...")
expl14  = shap.TreeExplainer(final14)
samp14  = Xte14.sample(min(300, len(Xte14)), random_state=42)
sv14    = expl14.shap_values(samp14)
if isinstance(sv14, np.ndarray) and sv14.ndim == 3:
    sv14 = [sv14[:, :, c] for c in range(sv14.shape[2])]
mean14  = sum(np.abs(s).mean(axis=0) for s in sv14)
shap14_df = pd.DataFrame({"feature": FEAT14, "mean_abs_shap": mean14})
shap14_df = shap14_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
print(f"  14일 SHAP Top-5: {shap14_df['feature'].head(5).tolist()}")

top20_14 = shap14_df.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_14["feature"][::-1], top20_14["mean_abs_shap"][::-1], color="#B71C1C")
ax.set_xlabel("평균 |SHAP| (전체 클래스 합산)")
ax.set_title("v2 14일 선행 모델 — 피처 중요도 Top-20 (전체 피처)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "plots" / "v2_shap_14day.png", dpi=150)
plt.close()

# 7일 vs 14일 혼동행렬 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
confusion_heatmap(yte_f.values, yp_final,
                  f"7일 선행 모델\n(boundary_recall={m_final['boundary_recall']})", axes[0])
confusion_heatmap(yte14.values, yp14,
                  f"14일 선행 모델\n(boundary_recall={m14['boundary_recall']})", axes[1])
plt.suptitle("7일 vs 14일 선행 예측 비교 (Test 2024-2025)", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "v2_7d_vs_14d_confusion.png", dpi=150)
plt.close()

# 14일 시계열 예측 플롯
print("  14일 지점별 시계열 플롯 저장...")
test14_plot = test14.copy()
test14_plot["pred_stage"]  = yp14
test14_plot["prob_normal"] = tp14[:, 0]
test14_plot["prob_caution"]= tp14[:, 1]
test14_plot["prob_alert"]  = tp14[:, 2]
test14_plot.to_csv(OUT / "reports" / "holdout_predictions_14d_v2.csv",
                   index=False, encoding="utf-8-sig")

for site in df14["채수위치"].unique():
    sub = test14_plot[test14_plot["채수위치"] == site].sort_values("조사일")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(sub["조사일"], sub["total_cyano"], color="#2196F3", lw=1.2, label="total_cyano")
    axes[0].axhline(1000,  color="orange", ls="--", lw=1, label="관심 기준")
    axes[0].axhline(10000, color="red",    ls="--", lw=1, label="경계 기준")
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_ylabel("유해남조류 (cells/mL)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{site} — 유해남조류 농도", fontsize=11)

    for _, row in sub.iterrows():
        axes[1].axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["target_d14"]), "gray"), alpha=0.45)
        axes[1].axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["pred_stage"]), "gray"), alpha=0.15)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color="green",  alpha=0.5, label="미발령"),
        Patch(color="orange", alpha=0.5, label="관심"),
        Patch(color="red",    alpha=0.5, label="경계"),
        Patch(color="gray",   alpha=0.2, label="예측(연함)"),
    ], loc="upper right", fontsize=8)
    axes[1].set_yticks([])
    axes[1].set_title("발령단계: 실제(진함) vs 예측(연함) — 14일 선행", fontsize=11)

    axes[2].stackplot(sub["조사일"],
                      sub["prob_normal"], sub["prob_caution"], sub["prob_alert"],
                      labels=["미발령", "관심", "경계"],
                      colors=["#66BB6A", "#FFA726", "#EF5350"], alpha=0.85)
    axes[2].set_ylabel("예측 확률")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("14일 선행 경보 발령 확률", fontsize=11)

    plt.suptitle(f"대청댐 {site} — 14일 선행 남조류 경보 예측 (전체 피처, Train 2016-2023)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "plots" / f"v2_timeseries_14d_{site}.png", dpi=150)
    plt.close()
    print(f"    저장: v2_timeseries_14d_{site}.png")

result_v2_14d = {
    "lead_days": 14,
    "n_features": len(FEAT14),
    "threshold_boundary": float(THR14_B),
    "threshold_caution": float(THR14_C),
    "metrics": m14,
    "shap_top5": shap14_df["feature"].head(5).tolist(),
}

# =============================================================================
# 최종 비교 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 비교 요약")
print("=" * 70)

# 이전 v1 결과 로드 (있으면)
v1_path = BASE.parent / "outputs" / "improvements" / "reports" / "improvement_summary.json"
v1_ref = None
if v1_path.exists():
    with open(v1_path) as f:
        v1_ref = json.load(f)

rows = [
    ("v1 기존 (macro-F1 임계값)",     {"boundary_recall": 0.1473, "alert_f1": 0.9315, "macro_f1": 0.5860}),
    ("v1 cost-optimal (0.05임계값)",  {"boundary_recall": 0.5241, "alert_f1": 0.9282, "macro_f1": 0.6631}),
    ("v2 수정1: 7일 최종모델",         m_final),
    ("v2 수정2: 14일 전체피처",        m14),
]

print(f"\n  {'항목':<35} | {'boundary_recall':>15} | {'alert_f1':>9} | {'macro_f1':>9}")
print("  " + "-" * 75)
for label, m in rows:
    br = m.get("boundary_recall") or 0.0
    print(f"  {label:<35} | {br:>15.4f} | {m['alert_f1']:>9.4f} | {m['macro_f1']:>9.4f}")

summary = {
    "split_strategy": {
        "tune_train": "2016-2022",
        "tune_val":   "2023 (경계 70건)",
        "final_train": "2016-2023 (경계 175건)",
        "test": "2024-2025 (경계 353건)",
    },
    "result_7day":  result_v2_split,
    "result_14day": result_v2_14d,
}
with open(OUT / "reports" / "improvement_v2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"\n  저장 완료: {OUT}")
print("\n완료!")
