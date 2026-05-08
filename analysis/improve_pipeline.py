"""
개선 파이프라인 — 5단계 순차 실행
====================================
Task 1: 경계 클래스 cost-optimal 임계값 탐색
Task 2: 지점별(site-specific) 분리 모델
Task 3: 외인성 피처 전용 14일 선행 모델
Task 4: Conformal Prediction (커버리지 90% 보장)
Task 5: LSTM + LightGBM 앙상블

실행: python3.10 improve_pipeline.py
결과: outputs/improvements/ 폴더
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
    accuracy_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap

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
BASE = Path(__file__).parent
DATA_PATH = BASE / "finaldata.csv"
OUT = BASE.parent / "outputs" / "improvements"
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

LEAD_DAYS   = 7
STAGE_MAP   = {"미발령": 0, "관심": 1, "경계": 2, "조류대발생": 3}
LABELS      = ["미발령", "관심", "경계"]
CYANO_COLS  = ["microcystis", "anabaena", "oscillatoria", "aphanizomenon"]

# ── 공통 데이터 준비 ──────────────────────────────────────────────────────────
def load_and_prep(data_path=DATA_PATH):
    df = pd.read_csv(data_path, parse_dates=["조사일"])
    df = df.sort_values(["채수위치", "조사일"]).reset_index(drop=True)
    df["stage_num"] = df["발령단계"].map(STAGE_MAP).fillna(0).astype(int)
    df = df.drop(columns=["일조시간 합계(hr)", "투명도"], errors="ignore")

    for site, g in df.groupby("채수위치"):
        idx = g.index
        df.loc[idx, CYANO_COLS] = g[CYANO_COLS].interpolate(method="linear", limit=14)
    df["total_cyano"] = df[CYANO_COLS].sum(axis=1)
    df["일강수량(mm)"] = df["일강수량(mm)"].fillna(df.get("강우량(mm)", 0))
    return df


def make_features(g: pd.DataFrame) -> pd.DataFrame:
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
    # 타겟
    g["target_d7"]    = g["stage_num"].shift(-LEAD_DAYS)
    g["target_binary"] = (g["target_d7"] >= 1).astype(float)
    return g


EXCLUDE = {"조사일", "채수위치", "발령단계", "stage_num",
           "target_d7", "target_binary", "일강수량(mm)"}

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in EXCLUDE and df[c].dtype != object]


def split_tvt(df: pd.DataFrame):
    train = df[df["조사일"] < "2023-01-01"]
    val   = df[(df["조사일"] >= "2023-01-01") & (df["조사일"] < "2024-01-01")]
    test  = df[df["조사일"] >= "2024-01-01"]
    return train, val, test


def lgb_params_base():
    return dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        n_estimators=1500, learning_rate=0.03, max_depth=7, num_leaves=127,
        min_child_samples=10, feature_fraction=0.75, bagging_fraction=0.75,
        bagging_freq=5, lambda_l1=0.05, lambda_l2=0.1,
        verbose=-1, random_state=42, n_jobs=-1,
    )


def sample_weights(y, boost_cls=2, boost=3.0):
    counts = pd.Series(y).value_counts().sort_index()
    base   = {k: len(y) / (len(counts) * v) for k, v in counts.items()}
    base[boost_cls] = base.get(boost_cls, 1.0) * boost
    return pd.Series(y).map(base).values


def cost_metric(TP, TN, FP, FN, w_fp=1.0, w_fn=3.0):
    return w_fp * FP + w_fn * FN


def eval_binary(y_true, y_pred):
    yb_t = (np.array(y_true) >= 1).astype(int)
    yb_p = (np.array(y_pred) >= 1).astype(int)
    return dict(
        accuracy=round(accuracy_score(y_true, y_pred), 4),
        macro_f1=round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        alert_recall=round(recall_score(yb_t, yb_p, zero_division=0), 4),
        alert_precision=round(precision_score(yb_t, yb_p, zero_division=0), 4),
        alert_f1=round(f1_score(yb_t, yb_p, zero_division=0), 4),
        boundary_recall=round(
            recall_score(y_true[np.array(y_true)==2], y_pred[np.array(y_true)==2],
                         labels=[2], average="micro", zero_division=0), 4
        ) if (np.array(y_true) == 2).sum() > 0 else None,
    )

# =============================================================================
# Task 1: 경계 클래스 cost-optimal 임계값 탐색
# =============================================================================
print("\n" + "=" * 70)
print("Task 1: 경계 클래스 cost-optimal 임계값 탐색")
print("=" * 70)

df_raw = load_and_prep()
parts = [make_features(g) for _, g in df_raw.groupby("채수위치")]
df = pd.concat(parts).sort_values(["조사일", "채수위치"]).reset_index(drop=True)
df = df.dropna(subset=["target_d7"])
df["target_d7"] = df["target_d7"].astype(int)

FEAT = get_feature_cols(df)
train_df, val_df, test_df = split_tvt(df)

X_train = train_df[FEAT]; y_train = train_df["target_d7"]
X_val   = val_df[FEAT];   y_val   = val_df["target_d7"]
X_test  = test_df[FEAT];  y_test  = test_df["target_d7"]

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42, sampling_strategy={2: 1000})
X_tr_sm, y_tr_sm = ros.fit_resample(X_train, y_train)

model_full = lgb.LGBMClassifier(**lgb_params_base())
model_full.fit(
    X_tr_sm, y_tr_sm,
    sample_weight=sample_weights(y_tr_sm),
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
)
print(f"  전체 모델 best_iter={model_full.best_iteration_}")

val_prob  = model_full.predict_proba(X_val)
test_prob = model_full.predict_proba(X_test)

# ── 1-A. 기존 방식: macro-F1 기준 그리드 서치 ──────────────────────────────
print("\n  [1-A] 기존: Macro-F1 최대화 임계값 탐색")
best_thr_a, best_thr_c, best_mf1 = 0.30, 0.35, 0.0
for ta in np.arange(0.05, 0.50, 0.01):
    for tc in np.arange(0.15, 0.55, 0.05):
        yp = np.zeros(len(y_val), dtype=int)
        yp[val_prob[:, 2] >= ta] = 2
        yp[(val_prob[:, 1] >= tc) & (yp < 2)] = 1
        f1 = f1_score(y_val, yp, average="macro", zero_division=0)
        if f1 > best_mf1:
            best_mf1, best_thr_a, best_thr_c = f1, ta, tc

yp_mf1 = np.zeros(len(y_test), dtype=int)
yp_mf1[test_prob[:, 2] >= best_thr_a] = 2
yp_mf1[(test_prob[:, 1] >= best_thr_c) & (yp_mf1 < 2)] = 1
m1a = eval_binary(y_test.values, yp_mf1)
print(f"    임계값: 경계={best_thr_a:.2f}, 관심={best_thr_c:.2f}  val-F1={best_mf1:.4f}")
print(f"    테스트: {m1a}")

# ── 1-B. 새 방식: cost-weighted 최소화 (FP:1, FN:3) ─────────────────────────
print("\n  [1-B] 신규: Cost-weighted 최소화 (FP:1, FN:3)")
best_cost, best_ta_cost, best_tc_cost = 1e9, 0.12, 0.35
cost_grid = []
for ta in np.arange(0.04, 0.40, 0.01):
    for tc in np.arange(0.10, 0.55, 0.05):
        yp = np.zeros(len(y_val), dtype=int)
        yp[val_prob[:, 2] >= ta] = 2
        yp[(val_prob[:, 1] >= tc) & (yp < 2)] = 1
        yb_t = (y_val.values >= 1).astype(int)
        yb_p = (yp >= 1).astype(int)
        FP = int(((yb_p==1) & (yb_t==0)).sum())
        FN = int(((yb_p==0) & (yb_t==1)).sum())
        cost = cost_metric(0, 0, FP, FN)
        b_mask = y_val.values == 2
        b_rec  = float((yp[b_mask] == 2).sum()) / max(b_mask.sum(), 1)
        cost_grid.append({"ta": ta, "tc": tc, "cost": cost,
                          "FP": FP, "FN": FN, "b_rec": round(b_rec, 3)})
        if cost < best_cost:
            best_cost, best_ta_cost, best_tc_cost = cost, ta, tc

yp_cost = np.zeros(len(y_test), dtype=int)
yp_cost[test_prob[:, 2] >= best_ta_cost] = 2
yp_cost[(test_prob[:, 1] >= best_tc_cost) & (yp_cost < 2)] = 1
m1b = eval_binary(y_test.values, yp_cost)
print(f"    임계값: 경계={best_ta_cost:.2f}, 관심={best_tc_cost:.2f}  val-cost={best_cost}")
print(f"    테스트: {m1b}")

# ── 1-C. 경계 Recall 70%+ 보장하는 최소 비용 임계값 ──────────────────────────
print("\n  [1-C] 신규: 경계 Recall ≥ 0.70 제약 하 최소 비용")
filtered = [r for r in cost_grid if r["b_rec"] >= 0.70]
if filtered:
    best_r = min(filtered, key=lambda x: x["cost"])
    ta_c, tc_c = best_r["ta"], best_r["tc"]
    yp_rc = np.zeros(len(y_test), dtype=int)
    yp_rc[test_prob[:, 2] >= ta_c] = 2
    yp_rc[(test_prob[:, 1] >= tc_c) & (yp_rc < 2)] = 1
    m1c = eval_binary(y_test.values, yp_rc)
    print(f"    임계값: 경계={ta_c:.2f}, 관심={tc_c:.2f}  val-b_rec={best_r['b_rec']:.2f}  val-cost={best_r['cost']}")
    print(f"    테스트: {m1c}")
else:
    print("    val에서 경계 Recall 70% 달성 임계값 없음")
    ta_c, tc_c = best_ta_cost, best_tc_cost
    yp_rc, m1c = yp_cost, m1b

# ── 임계값 비용곡선 플롯 ──────────────────────────────────────────────────────
cg = pd.DataFrame(cost_grid)
pivot = cg.groupby("ta")["cost"].min().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(pivot["ta"], pivot["cost"], "b-o", ms=4)
axes[0].axvline(best_ta_cost, color="red", ls="--", label=f"best cost (ta={best_ta_cost:.2f})")
if filtered:
    axes[0].axvline(ta_c, color="green", ls="--", label=f"Recall≥70% (ta={ta_c:.2f})")
axes[0].set_xlabel("경계 임계값 (ta)")
axes[0].set_ylabel("val cost (FP+3*FN)")
axes[0].set_title("임계값 vs 비용")
axes[0].legend()
# 경계 Recall vs 비용 산점도
axes[1].scatter(cg["b_rec"], cg["cost"], c=cg["ta"], cmap="plasma", s=15, alpha=0.5)
axes[1].axvline(0.70, color="green", ls="--", label="Recall 70% 기준")
axes[1].set_xlabel("경계 val Recall")
axes[1].set_ylabel("val cost")
axes[1].set_title("경계 Recall vs 비용")
axes[1].legend()
plt.suptitle("Task 1: 임계값 최적화", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "task1_threshold_optimization.png", dpi=150)
plt.close()

task1_result = {
    "macro_f1_optimal": {"thr_boundary": float(best_thr_a), "thr_caution": float(best_thr_c),
                          "metrics": m1a},
    "cost_optimal":     {"thr_boundary": float(best_ta_cost), "thr_caution": float(best_tc_cost),
                          "metrics": m1b},
    "recall70_constrained": {"thr_boundary": float(ta_c), "thr_caution": float(tc_c),
                              "metrics": m1c},
}
print("\n  [요약 비교]")
for k, v in task1_result.items():
    print(f"    {k}: 경계={v['thr_boundary']:.2f} → boundary_recall={v['metrics']['boundary_recall']} alert_f1={v['metrics']['alert_f1']}")

# ── 최적 임계값 결정 (경계 Recall 70% 제약) ──────────────────────────────────
BEST_THR_BOUNDARY = ta_c
BEST_THR_CAUTION  = tc_c
print(f"\n  >> 이후 파이프라인 사용 임계값: 경계={BEST_THR_BOUNDARY:.2f}, 관심={BEST_THR_CAUTION:.2f}")

# =============================================================================
# Task 2: 지점별(site-specific) 분리 모델
# =============================================================================
print("\n" + "=" * 70)
print("Task 2: 지점별(site-specific) 분리 모델")
print("=" * 70)

site_results = {}
site_preds   = {}

for site in df["채수위치"].unique():
    print(f"\n  ── {site} ──")
    sub = df[df["채수위치"] == site].copy()
    f_cols = get_feature_cols(sub)

    tr = sub[sub["조사일"] < "2023-01-01"]
    vl = sub[(sub["조사일"] >= "2023-01-01") & (sub["조사일"] < "2024-01-01")]
    te = sub[sub["조사일"] >= "2024-01-01"]

    Xtr, ytr = tr[f_cols], tr["target_d7"]
    Xvl, yvl = vl[f_cols], vl["target_d7"]
    Xte, yte = te[f_cols], te["target_d7"]

    # 경계 클래스가 없으면 오버샘플 불가 → 단순 가중치로 대체
    n_boundary = (ytr == 2).sum()
    if n_boundary >= 5:
        target_sm = min(max(n_boundary * 3, 200), 800)
        ros_s = RandomOverSampler(random_state=42, sampling_strategy={2: target_sm})
        Xtr_sm, ytr_sm = ros_s.fit_resample(Xtr, ytr)
    else:
        Xtr_sm, ytr_sm = Xtr, ytr

    params_s = lgb_params_base()
    params_s.update(n_estimators=1200, min_child_samples=5)
    m_site = lgb.LGBMClassifier(**params_s)
    m_site.fit(
        Xtr_sm, ytr_sm,
        sample_weight=sample_weights(ytr_sm),
        eval_set=[(Xvl, yvl)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )

    prob_te  = m_site.predict_proba(Xte)
    yp_site  = np.zeros(len(yte), dtype=int)
    yp_site[prob_te[:, 2] >= BEST_THR_BOUNDARY] = 2
    yp_site[(prob_te[:, 1] >= BEST_THR_CAUTION) & (yp_site < 2)] = 1

    m = eval_binary(yte.values, yp_site)
    site_results[site] = m
    site_preds[site]   = (yte.values, yp_site, prob_te, te["조사일"].values)
    print(f"    iter={m_site.best_iteration_}  {m}")

# ── 지점별 혼동행렬 ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(site_results), figsize=(5*len(site_results), 4))
if len(site_results) == 1:
    axes = [axes]
for ax, (site, (yt, yp, _, _)) in zip(axes, site_preds.items()):
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    import seaborn as sns
    sns.heatmap(pd.DataFrame(cm, index=LABELS, columns=LABELS),
                annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"{site}\n경계Recall={site_results[site]['boundary_recall']}")
    ax.set_ylabel("실제"); ax.set_xlabel("예측")
plt.suptitle("Task 2: 지점별 분리 모델 혼동행렬", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "task2_site_confusion.png", dpi=150)
plt.close()

# 전체 vs 지점별 비교
print("\n  [전체 모델 vs 지점별 모델 비교]")
all_yt, all_yp = [], []
for yt, yp, _, _ in site_preds.values():
    all_yt.extend(yt); all_yp.extend(yp)
m_site_agg = eval_binary(np.array(all_yt), np.array(all_yp))
m_full_ref  = eval_binary(y_test.values, yp_rc)
print(f"    전체모델(Recall70제약): {m_full_ref}")
print(f"    지점별모델(집계):       {m_site_agg}")

task2_result = {
    "site_metrics": site_results,
    "aggregated":   m_site_agg,
    "full_model_ref": m_full_ref,
}

# =============================================================================
# Task 3: 외인성 피처 전용 14일 선행 모델
# =============================================================================
print("\n" + "=" * 70)
print("Task 3: 외인성 피처 전용 14일 선행 모델")
print("=" * 70)

LEAD_LONG = 14
EXOG_KEYWORDS = ["기온", "기상", "강수", "일사", "풍속", "습도", "전운",
                 "수위", "저수", "유입", "방류", "gdd", "month", "sin_", "cos_",
                 "수온", "pH", "DO", "탁도"]
EXOG_CYANO_EXCL = ["total_cyano", "microcystis", "anabaena",
                   "oscillatoria", "aphanizomenon", "Chl-a", "stage_lag"]

def is_exog(col: str) -> bool:
    if any(kw in col for kw in EXOG_CYANO_EXCL):
        return False
    return any(kw in col for kw in EXOG_KEYWORDS)

# 14일 타겟 생성
parts14 = []
for site, g in df_raw.groupby("채수위치"):
    gf = make_features(g)
    gf["target_d14"] = g["stage_num"].reindex(gf.index).shift(-LEAD_LONG)
    parts14.append(gf)
df14 = pd.concat(parts14).sort_values(["조사일", "채수위치"]).reset_index(drop=True)
df14 = df14.dropna(subset=["target_d14"])
df14["target_d14"] = df14["target_d14"].astype(int).clip(upper=2)

ALL_FEAT = get_feature_cols(df14)
EXOG_FEAT = [c for c in ALL_FEAT if is_exog(c)]
print(f"  외인성 피처 수: {len(EXOG_FEAT)} / 전체 {len(ALL_FEAT)}")

tr14 = df14[df14["조사일"] < "2023-01-01"]
vl14 = df14[(df14["조사일"] >= "2023-01-01") & (df14["조사일"] < "2024-01-01")]
te14 = df14[df14["조사일"] >= "2024-01-01"]

Xtr14, ytr14 = tr14[EXOG_FEAT], tr14["target_d14"]
Xvl14, yvl14 = vl14[EXOG_FEAT], vl14["target_d14"]
Xte14, yte14 = te14[EXOG_FEAT], te14["target_d14"]

# 오버샘플
n_b = (ytr14 == 2).sum()
if n_b >= 5:
    ros14 = RandomOverSampler(random_state=42,
                              sampling_strategy={2: min(n_b * 3, 1000)})
    Xtr14_sm, ytr14_sm = ros14.fit_resample(Xtr14, ytr14)
else:
    Xtr14_sm, ytr14_sm = Xtr14, ytr14

params14 = lgb_params_base()
params14.update(n_estimators=1200, learning_rate=0.05, max_depth=6, num_leaves=63)
model14 = lgb.LGBMClassifier(**params14)
model14.fit(
    Xtr14_sm, ytr14_sm,
    sample_weight=sample_weights(ytr14_sm),
    eval_set=[(Xvl14, yvl14)],
    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
)
print(f"  14일 모델 best_iter={model14.best_iteration_}")

prob14    = model14.predict_proba(Xte14)
yp14      = np.zeros(len(yte14), dtype=int)
yp14[prob14[:, 2] >= BEST_THR_BOUNDARY] = 2
yp14[(prob14[:, 1] >= BEST_THR_CAUTION) & (yp14 < 2)] = 1

m14 = eval_binary(yte14.values, yp14)
print(f"  14일 외인성 전용 테스트: {m14}")

# 리드타임 1/7/14 비교
leadtime_compare = {
    "7일_전체피처":    m_full_ref,
    "14일_외인성피처": m14,
}

# SHAP (14일 모델)
expl14 = shap.TreeExplainer(model14)
shap_s  = Xte14.sample(min(300, len(Xte14)), random_state=42)
sv14    = expl14.shap_values(shap_s)
if isinstance(sv14, np.ndarray) and sv14.ndim == 3:
    sv14 = [sv14[:, :, c] for c in range(sv14.shape[2])]
mean14  = sum(np.abs(s).mean(axis=0) for s in sv14)
shap14_df = pd.DataFrame({"feature": EXOG_FEAT, "mean_abs_shap": mean14})
shap14_df = shap14_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

top15 = shap14_df.head(15)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1], color="#C62828")
ax.set_xlabel("평균 |SHAP| (14일 외인성 모델)")
ax.set_title("Task 3: 14일 선행 외인성 피처 중요도", fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "plots" / "task3_shap_exog14.png", dpi=150)
plt.close()
print(f"  SHAP Top-5: {shap14_df['feature'].head(5).tolist()}")

task3_result = {"lead_days": LEAD_LONG, "n_exog_features": len(EXOG_FEAT),
                "metrics": m14, "shap_top5": shap14_df["feature"].head(5).tolist()}

# =============================================================================
# Task 4: Conformal Prediction (Split CP, 90% 커버리지 보장)
# =============================================================================
print("\n" + "=" * 70)
print("Task 4: Conformal Prediction — 경보 확률 90% 보장 구간")
print("=" * 70)

# Split CP: val set를 캘리브레이션으로 사용
# 이진 확률 p(alert≥1) 에 대해 CP interval 구성
# nonconformity score: s = 1 - p_hat(true_class)
val_prob_full   = model_full.predict_proba(X_val)
test_prob_full  = model_full.predict_proba(X_test)

y_val_bin   = (y_val.values >= 1).astype(int)
y_test_bin  = (y_test.values >= 1).astype(int)

# p(alert) = p(관심) + p(경계)
p_alert_val  = val_prob_full[:, 1] + val_prob_full[:, 2]
p_alert_test = test_prob_full[:, 1] + test_prob_full[:, 2]

# nonconformity score: 실제 클래스의 확률을 1에서 뺀 값
nc_scores = np.where(y_val_bin == 1,
                     1 - p_alert_val,    # alert: 낮을수록 이상
                     p_alert_val)        # normal: 높을수록 이상

alpha = 0.10  # 90% coverage 목표
n_cal = len(nc_scores)
quantile_cp = np.quantile(nc_scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
print(f"  보정 분위수 q={quantile_cp:.4f}  (캘리브레이션 n={n_cal})")

# 예측 집합: p_alert >= 1 - q → alert 포함
cp_includes_alert = (p_alert_test >= 1 - quantile_cp).astype(int)

# 커버리지 계산
coverage = float((cp_includes_alert[y_test_bin == 1] == 1).mean()) if y_test_bin.sum() > 0 else 0.0
tn_coverage = float((cp_includes_alert[y_test_bin == 0] == 0).mean()) if (y_test_bin == 0).sum() > 0 else 0.0
efficiency  = float(1 - cp_includes_alert.mean())  # 구간이 좁을수록 높음

print(f"  경보 커버리지(Recall): {coverage:.4f}  (목표 0.90)")
print(f"  정상 특이도:            {tn_coverage:.4f}")
print(f"  효율성(예측집합 공집합 비율): {efficiency:.4f}")

# 임계값별 커버리지 vs precision 트레이드오프
alpha_grid = np.arange(0.05, 0.50, 0.05)
cp_tradeoff = []
for a in alpha_grid:
    q = np.quantile(nc_scores, np.ceil((n_cal + 1) * (1 - a)) / n_cal)
    inc = (p_alert_test >= 1 - q).astype(int)
    cov = float((inc[y_test_bin == 1] == 1).mean()) if y_test_bin.sum() > 0 else 0.0
    spec = float((inc[y_test_bin == 0] == 0).mean()) if (y_test_bin == 0).sum() > 0 else 0.0
    cp_tradeoff.append({"alpha": round(a, 2), "coverage": round(cov, 3), "specificity": round(spec, 3)})

cp_df = pd.DataFrame(cp_tradeoff)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(cp_df["alpha"], cp_df["coverage"], "b-o", ms=5, label="경보 Coverage")
ax.plot(cp_df["alpha"], cp_df["specificity"], "r-s", ms=5, label="정상 Specificity")
ax.axhline(0.90, color="blue", ls="--", lw=1, label="90% 목표")
ax.axvline(alpha, color="gray", ls="--", lw=1, label=f"α={alpha}")
ax.set_xlabel("α (오류율)")
ax.set_ylabel("비율")
ax.set_title("Task 4: Conformal Prediction — α별 Coverage & Specificity")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "plots" / "task4_conformal_coverage.png", dpi=150)
plt.close()

# 비교: 기존 Quantile 커버리지 52% → CP 보정 후
print(f"\n  [비교] 기존 Quantile 구간 커버리지: 52.2%  →  CP(α=0.10): {coverage*100:.1f}%")

task4_result = {
    "alpha": alpha,
    "calibration_quantile": float(quantile_cp),
    "test_alert_coverage": round(coverage, 4),
    "test_normal_specificity": round(tn_coverage, 4),
    "alpha_tradeoff": cp_df.to_dict(orient="records"),
}

# =============================================================================
# Task 5: LSTM + LightGBM 앙상블
# =============================================================================
print("\n" + "=" * 70)
print("Task 5: LSTM + LightGBM 앙상블")
print("=" * 70)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

# ── LSTM 입력 구성: 지점별 시계열로 슬라이딩 윈도우 ─────────────────────────
SEQ_LEN = 8   # 과거 8주 → 7일 후 예측

def make_lstm_dataset(sub: pd.DataFrame, feat_cols: list, seq_len: int):
    sub = sub.sort_values("조사일").reset_index(drop=True)
    sub_filled = sub[feat_cols].fillna(method="ffill").fillna(0)
    X_seq, y_seq, dates = [], [], []
    for i in range(seq_len, len(sub)):
        row_y = sub["target_d7"].iloc[i]
        if pd.isna(row_y):
            continue
        X_seq.append(sub_filled.values[i-seq_len:i])
        y_seq.append(int(row_y))
        dates.append(sub["조사일"].iloc[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32), dates

# 피처 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

FEAT_LSTM = FEAT[:60]  # 너무 많으면 LSTM 과적합 → 상위 60개만
# 훈련 데이터로 fit
X_train_sc = scaler.fit_transform(X_train[FEAT_LSTM].fillna(0))

df_lstm = df.copy()
df_lstm[FEAT_LSTM] = scaler.transform(df[FEAT_LSTM].fillna(0))

lstm_Xtr, lstm_ytr, lstm_dtr = [], [], []
lstm_Xvl, lstm_yvl, lstm_dvl = [], [], []
lstm_Xte, lstm_yte, lstm_dte = [], [], []

for site, g in df_lstm.groupby("채수위치"):
    tr_s = g[g["조사일"] < "2023-01-01"]
    vl_s = g[(g["조사일"] >= "2023-01-01") & (g["조사일"] < "2024-01-01")]
    te_s = g[g["조사일"] >= "2024-01-01"]
    for subset, Xlist, ylist, dlist in [(tr_s, lstm_Xtr, lstm_ytr, lstm_dtr),
                                         (vl_s, lstm_Xvl, lstm_yvl, lstm_dvl),
                                         (te_s, lstm_Xte, lstm_yte, lstm_dte)]:
        Xs, ys, ds = make_lstm_dataset(subset, FEAT_LSTM, SEQ_LEN)
        Xlist.append(Xs); ylist.extend(ys); dlist.extend(ds)

lstm_Xtr = np.concatenate(lstm_Xtr, axis=0)
lstm_Xvl = np.concatenate(lstm_Xvl, axis=0)
lstm_Xte = np.concatenate(lstm_Xte, axis=0)
lstm_ytr = np.array(lstm_ytr); lstm_yvl = np.array(lstm_yvl); lstm_yte = np.array(lstm_yte)

print(f"  LSTM 데이터: train={lstm_Xtr.shape}, val={lstm_Xvl.shape}, test={lstm_Xte.shape}")

# 클래스 가중치
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=lstm_ytr)
cw[2] *= 3.0   # 경계 추가 강조
class_weight_dict = {i: float(w) for i, w in enumerate(cw)}
print(f"  클래스 가중치: {class_weight_dict}")

# ── LSTM 모델 구성 ─────────────────────────────────────────────────────────────
def build_lstm(seq_len, n_feat, n_class=3):
    inp = keras.Input(shape=(seq_len, n_feat))
    x   = layers.LSTM(64, return_sequences=True)(inp)
    x   = layers.Dropout(0.3)(x)
    x   = layers.LSTM(32)(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_class, activation="softmax")(x)
    m   = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

lstm_model = build_lstm(SEQ_LEN, len(FEAT_LSTM))
lstm_model.summary(print_fn=lambda x: None)

cb = [
    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, verbose=0),
]
history = lstm_model.fit(
    lstm_Xtr, lstm_ytr,
    validation_data=(lstm_Xvl, lstm_yvl),
    epochs=80, batch_size=64,
    class_weight=class_weight_dict,
    callbacks=cb, verbose=0,
)
print(f"  LSTM 학습 완료 (epochs={len(history.history['loss'])})")

# LSTM 단독 성능
lstm_prob_te  = lstm_model.predict(lstm_Xte, verbose=0)
lstm_yp       = np.zeros(len(lstm_yte), dtype=int)
lstm_yp[lstm_prob_te[:, 2] >= BEST_THR_BOUNDARY] = 2
lstm_yp[(lstm_prob_te[:, 1] >= BEST_THR_CAUTION) & (lstm_yp < 2)] = 1
m_lstm = eval_binary(lstm_yte, lstm_yp)
print(f"  LSTM 단독: {m_lstm}")

# ── LGB 확률 정렬 (LSTM과 동일 인덱스로) ──────────────────────────────────────
# LSTM 테스트 날짜와 LGB 테스트 날짜 맞추기
te_dates_lgb  = test_df["조사일"].values
te_dates_lstm = np.array(lstm_dte)

# LSTM 날짜로 LGB 확률 필터
lgb_prob_aligned = []
for d in te_dates_lstm:
    mask = te_dates_lgb == d
    if mask.sum() > 0:
        # 여러 지점이 있을 수 있으므로 평균
        lgb_prob_aligned.append(test_prob_full[mask].mean(axis=0))
    else:
        lgb_prob_aligned.append(np.array([1/3, 1/3, 1/3]))
lgb_prob_aligned = np.array(lgb_prob_aligned)

# ── 앙상블: LGB 0.6 + LSTM 0.4 ──────────────────────────────────────────────
for w_lgb in [0.5, 0.6, 0.7]:
    w_lstm = 1 - w_lgb
    ens_prob = w_lgb * lgb_prob_aligned + w_lstm * lstm_prob_te
    yp_ens   = np.zeros(len(lstm_yte), dtype=int)
    yp_ens[ens_prob[:, 2] >= BEST_THR_BOUNDARY] = 2
    yp_ens[(ens_prob[:, 1] >= BEST_THR_CAUTION) & (yp_ens < 2)] = 1
    m_ens = eval_binary(lstm_yte, yp_ens)
    print(f"  앙상블 LGB{w_lgb:.0%}+LSTM{w_lstm:.0%}: {m_ens}")

# 최적 가중치 탐색 (val 기준)
best_w_lgb, best_ens_f1 = 0.6, 0.0
lstm_prob_vl = lstm_model.predict(lstm_Xvl, verbose=0)
vl_dates_lgb  = val_df["조사일"].values
vl_dates_lstm = np.array(lstm_dvl)
lgb_vl_aligned = []
for d in vl_dates_lstm:
    mask = vl_dates_lgb == d
    if mask.sum() > 0:
        lgb_vl_aligned.append(val_prob_full[mask].mean(axis=0))
    else:
        lgb_vl_aligned.append(np.array([1/3, 1/3, 1/3]))
lgb_vl_aligned = np.array(lgb_vl_aligned)

for w in np.arange(0.3, 0.9, 0.05):
    ep = w * lgb_vl_aligned + (1 - w) * lstm_prob_vl
    yp_e = np.zeros(len(lstm_yvl), dtype=int)
    yp_e[ep[:, 2] >= BEST_THR_BOUNDARY] = 2
    yp_e[(ep[:, 1] >= BEST_THR_CAUTION) & (yp_e < 2)] = 1
    f1e = f1_score(lstm_yvl, yp_e, average="macro", zero_division=0)
    if f1e > best_ens_f1:
        best_ens_f1, best_w_lgb = f1e, w

print(f"\n  최적 앙상블 가중치: LGB={best_w_lgb:.2f}, LSTM={1-best_w_lgb:.2f}  (val macro-F1={best_ens_f1:.4f})")
ens_prob_best = best_w_lgb * lgb_prob_aligned + (1 - best_w_lgb) * lstm_prob_te
yp_ens_best   = np.zeros(len(lstm_yte), dtype=int)
yp_ens_best[ens_prob_best[:, 2] >= BEST_THR_BOUNDARY] = 2
yp_ens_best[(ens_prob_best[:, 1] >= BEST_THR_CAUTION) & (yp_ens_best < 2)] = 1
m_ens_best = eval_binary(lstm_yte, yp_ens_best)
print(f"  최적 앙상블 테스트: {m_ens_best}")

# 학습 곡선
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"],     label="train loss")
ax.plot(history.history["val_loss"], label="val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Task 5: LSTM 학습 곡선")
ax.legend(); plt.tight_layout()
fig.savefig(OUT / "plots" / "task5_lstm_loss.png", dpi=150)
plt.close()

task5_result = {
    "lstm_only":         m_lstm,
    "ensemble_optimal":  m_ens_best,
    "best_lgb_weight":   round(float(best_w_lgb), 2),
}

# =============================================================================
# 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약 — 개선 전후 비교")
print("=" * 70)

summary = {
    "task1_threshold": task1_result,
    "task2_site_model": task2_result,
    "task3_exog14": task3_result,
    "task4_conformal": task4_result,
    "task5_lstm_ensemble": task5_result,
}

# 핵심 지표 비교표
print("\n  항목                        | boundary_recall | alert_f1  | macro_f1")
print("  " + "-" * 70)
rows = [
    ("기존(macro-F1 임계값)",         m1a),
    ("Task1: cost-최적 임계값",       m1b),
    ("Task1: Recall70% 제약",         m1c),
    ("Task2: 지점별 집계",            m_site_agg),
    ("Task3: 외인성14일",             m14),
    ("Task5: LSTM단독",               m_lstm),
    ("Task5: LGB+LSTM 앙상블",        m_ens_best),
]
for label, m in rows:
    br = m.get("boundary_recall") or 0.0
    print(f"  {label:<30} | {br:<15.4f} | {m['alert_f1']:<9.4f} | {m['macro_f1']:.4f}")

with open(OUT / "reports" / "improvement_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n  결과 저장: {OUT}/reports/improvement_summary.json")
print(f"  플롯 저장: {OUT}/plots/")
print("\n완료!")
