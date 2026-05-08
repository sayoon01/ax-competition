"""
대청댐 유해남조류 경보 발령 예측 AI 모델
=========================================
목표:
  1) 7일 선행 경보 발령 단계(미발령/관심/경계) 예측
  2) 녹조 발생 주요 영향 인자 도출 (SHAP)
  3) 시나리오 기반 의사결정 지원

데이터 출처:
  - 수질/조류: 환경부 물환경정보시스템 (water.nier.go.kr)
  - 기상: 기상청 기상자료개방포털 (data.kma.go.kr)
  - 댐 운영: K-water 대청댐 운영 데이터

경보 기준 (조류경보제 운영지침, 기후에너지환경부):
  - 관심: 유해남조류 1,000 cells/mL 이상 (2회 연속 초과)
  - 경계: 유해남조류 10,000 cells/mL 이상 (2회 연속 초과)
  - 조류대발생: 유해남조류 1,000,000 cells/mL 이상 (2회 연속 초과)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
import lightgbm as lgb
import shap
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            fe = fm.FontEntry(fname=p, name="KorFont")
            fm.fontManager.ttflist.append(fe)
            plt.rcParams["font.family"] = "KorFont"
            return
    plt.rcParams["font.family"] = "DejaVu Sans"

set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA_PATH = BASE / "finaldata.csv"
OUT = BASE / "outputs"
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

LEAD_DAYS = 7   # 선행 예측 일수

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────
print("=" * 60)
print("[1] 데이터 로드 및 기본 탐색")
print("=" * 60)

df_raw = pd.read_csv(DATA_PATH, parse_dates=["조사일"])
df_raw = df_raw.sort_values(["채수위치", "조사일"]).reset_index(drop=True)

print(f"  원본 데이터: {df_raw.shape[0]:,}행 × {df_raw.shape[1]}열")
print(f"  기간: {df_raw['조사일'].min().date()} ~ {df_raw['조사일'].max().date()}")
print(f"  모니터링 지점: {df_raw['채수위치'].unique().tolist()}")
print(f"  발령단계 분포:\n{df_raw['발령단계'].value_counts()}")

# ── 2. 전처리 ───────────────────────────────────────────────────────────────
print("\n[2] 데이터 전처리")

# 발령단계 수치 인코딩 (경보 심각도 순)
STAGE_MAP = {"미발령": 0, "관심": 1, "경계": 2, "조류대발생": 3}
df_raw["stage_num"] = df_raw["발령단계"].map(STAGE_MAP).fillna(0).astype(int)

# 불필요 컬럼 제거 (결측 90%+ 이상)
DROP_COLS = ["일조시간 합계(hr)", "투명도"]
df_raw = df_raw.drop(columns=DROP_COLS, errors="ignore")

# total_cyano 결측 → 4종 합계로 보완, 그래도 없으면 선형보간
cyano_cols = ["microcystis", "anabaena", "oscillatoria", "aphanizomenon"]
for site, g in df_raw.groupby("채수위치"):
    idx = g.index
    df_raw.loc[idx, cyano_cols] = g[cyano_cols].interpolate(method="linear", limit=14)
df_raw["total_cyano"] = df_raw[cyano_cols].sum(axis=1)

# 일강수량: 강우량 컬럼으로 채우기
df_raw["일강수량(mm)"] = df_raw["일강수량(mm)"].fillna(df_raw["강우량(mm)"])

# ── 3. 피처 엔지니어링 (지점별 시계열 순서 유지) ───────────────────────────
print("[3] 피처 엔지니어링")

def make_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("조사일")

    # --- 생물 지표 ---
    for col in ["total_cyano"] + cyano_cols + ["Chl-a (㎎/㎥)"]:
        for lag in [1, 3, 7, 14]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        for win in [3, 7, 14]:
            g[f"{col}_roll{win}m"] = g[col].shift(1).rolling(win).mean()
            g[f"{col}_roll{win}max"] = g[col].shift(1).rolling(win).max()

    # --- 수질 지표 ---
    for col in ["수온(℃)", "pH", "DO(㎎/L)", "탁도"]:
        for lag in [1, 3, 7]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()

    # --- 기상 지표 ---
    for col in ["평균기온(°C)", "최고기온(°C)", "합계 일사량(MJ/m2)",
                "평균 풍속(m/s)", "평균 상대습도(%)"]:
        for lag in [1, 3, 7]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()

    # --- 강수량 ---
    for col in ["일강수량(mm)", "강우량(mm)"]:
        for lag in [1, 3, 7, 14]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7sum"] = g[col].shift(1).rolling(7).sum()
        g[f"{col}_roll14sum"] = g[col].shift(1).rolling(14).sum()

    # --- 댐 운영 ---
    for col in ["수위(EL.m)", "저수율(%)", "유입량(㎥/s)", "총방류량(㎥/s)"]:
        for lag in [1, 3, 7]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()

    # --- 누적 온도 (성장 열량 proxy) ---
    g["gdd7"] = g["최고기온(°C)"].shift(1).clip(lower=10).rolling(7).sum()   # 최고기온 ≥10°C 7일 누적
    g["gdd14"] = g["최고기온(°C)"].shift(1).clip(lower=10).rolling(14).sum()

    # --- 수온-기온 차 (성층 지표) ---
    g["temp_diff"] = g["수온(℃)"] - g["평균기온(°C)"]
    g["temp_diff_lag1"] = g["temp_diff"].shift(1)
    g["temp_diff_roll7m"] = g["temp_diff"].shift(1).rolling(7).mean()

    # --- 발령 이력 (직전 경보 단계) ---
    for lag in [1, 3, 7]:
        g[f"stage_lag{lag}"] = g["stage_num"].shift(lag)
    g["stage_roll7max"] = g["stage_num"].shift(1).rolling(7).max()

    # --- 계절성 ---
    g["month"] = g["조사일"].dt.month
    g["week_of_year"] = g["조사일"].dt.isocalendar().week.astype(int)
    g["sin_month"] = np.sin(2 * np.pi * g["month"] / 12)
    g["cos_month"] = np.cos(2 * np.pi * g["month"] / 12)

    # --- 7일 선행 타겟 ---
    g["target_d7"] = g["stage_num"].shift(-LEAD_DAYS)   # 7일 후 단계
    g["target_binary"] = (g["target_d7"] >= 1).astype(float)  # 관심 이상 여부

    return g

parts = []
for site, grp in df_raw.groupby("채수위치"):
    parts.append(make_features(grp))
df = pd.concat(parts).sort_values(["조사일", "채수위치"]).reset_index(drop=True)

# 타겟 결측 제거 (미래 7일이 없는 마지막 행)
df = df.dropna(subset=["target_d7"])
df["target_d7"] = df["target_d7"].astype(int)

print(f"  피처 생성 후: {df.shape[0]:,}행 × {df.shape[1]}열")
print(f"  7일 선행 타겟 분포:\n{df['target_d7'].value_counts().sort_index()}")

# ── 4. 피처 선택 ────────────────────────────────────────────────────────────
EXCLUDE = {"조사일", "채수위치", "발령단계", "stage_num",
           "target_d7", "target_binary", "일강수량(mm)"}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE
                and df[c].dtype != object]

print(f"\n  사용 피처 수: {len(FEATURE_COLS)}")

# ── 5. 시계열 Train / Val / Test 분리 ──────────────────────────────────────
print("\n[4] 시계열 Train / Validation / Test 분리")

# Train: ~2022, Val: 2023, Test: 2024~
train_mask = df["조사일"] < "2023-01-01"
val_mask   = (df["조사일"] >= "2023-01-01") & (df["조사일"] < "2024-01-01")
test_mask  = df["조사일"] >= "2024-01-01"

X_train = df.loc[train_mask, FEATURE_COLS]
y_train = df.loc[train_mask, "target_d7"]
X_val   = df.loc[val_mask, FEATURE_COLS]
y_val   = df.loc[val_mask, "target_d7"]
X_test  = df.loc[test_mask, FEATURE_COLS]
y_test  = df.loc[test_mask, "target_d7"]

print(f"  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
print(f"  Test 기간: {df.loc[test_mask,'조사일'].min().date()} ~ {df.loc[test_mask,'조사일'].max().date()}")

# ── 5b. SMOTE — 경계 클래스 오버샘플링 (훈련 데이터만) ─────────────────────
print("  RandomOverSampler 오버샘플링 (경계 클래스 → 1,000개, 실제 샘플 복제)...")
ros = RandomOverSampler(random_state=42, sampling_strategy={2: 1000})
X_train_sm, y_train_sm = ros.fit_resample(X_train, y_train)
dist_sm = pd.Series(y_train_sm).value_counts().sort_index()
print(f"  오버샘플링 후 훈련 분포: { {int(k): int(v) for k, v in dist_sm.items()} }")

# ── 6. LightGBM 모델 학습 ───────────────────────────────────────────────────
print("\n[5] LightGBM 다중분류 모델 학습 (7일 선행 예측)")

# 클래스 가중치 – 오버샘플링 후에도 경계에 추가 강조 (3배)
class_counts_sm = pd.Series(y_train_sm).value_counts().sort_index()
base_w_sm = {k: len(y_train_sm) / (len(class_counts_sm) * v) for k, v in class_counts_sm.items()}
ALERT_WEIGHT_BOOST = 3.0
custom_w = {0: base_w_sm[0], 1: base_w_sm[1], 2: base_w_sm.get(2, 1.0) * ALERT_WEIGHT_BOOST}
sample_weight = pd.Series(y_train_sm).map(custom_w).values

params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 7,
    "num_leaves": 127,
    "min_child_samples": 10,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.75,
    "bagging_freq": 5,
    "lambda_l1": 0.05,
    "lambda_l2": 0.1,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train_sm, y_train_sm,
    sample_weight=sample_weight,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=80, verbose=False),
        lgb.log_evaluation(period=-1),
    ],
)

print(f"  Best iteration: {model.best_iteration_}")

# ── 모델 파일 저장 ──────────────────────────────────────────────────────────
model.booster_.save_model(str(OUT / "models" / "lgb_cyano_alert.txt"))
joblib.dump(model, OUT / "models" / "lgb_cyano_alert.pkl")
print(f"  모델 저장 완료: models/lgb_cyano_alert.txt + .pkl")

# ── 7. 평가 ─────────────────────────────────────────────────────────────────
print("\n[6] 모델 평가 (Test 세트: 2024~)")

y_pred_prob = model.predict_proba(X_test)

labels = ["미발령", "관심", "경계"]

# ── 임계값 최적화 (검증 세트 기준) ──
# 경계 클래스 확률이 임계값 이상이면 경계로 예측 (Macro-F1 최대화)
val_prob = model.predict_proba(X_val)
best_thr_alert, best_thr_caution, best_f1 = 0.30, 0.35, 0.0
for thr_a in np.arange(0.08, 0.50, 0.02):
    for thr_c in np.arange(0.20, 0.55, 0.05):
        pred_tmp = np.zeros(len(y_val), dtype=int)
        pred_tmp[val_prob[:, 2] >= thr_a] = 2
        mask_c = (val_prob[:, 1] >= thr_c) & (pred_tmp < 2)
        pred_tmp[mask_c] = 1
        f1_tmp = f1_score(y_val, pred_tmp, average="macro", zero_division=0)
        if f1_tmp > best_f1:
            best_f1, best_thr_alert, best_thr_caution = f1_tmp, thr_a, thr_c

print(f"  최적 임계값: 경계={best_thr_alert:.2f}, 관심={best_thr_caution:.2f} (val Macro-F1={best_f1:.4f})")

# 최적 임계값 적용 (경계 우선, 관심 차순)
y_pred = np.zeros(len(y_test), dtype=int)
y_pred[y_pred_prob[:, 2] >= best_thr_alert] = 2
mask_c = (y_pred_prob[:, 1] >= best_thr_caution) & (y_pred < 2)
y_pred[mask_c] = 1

print("\n  분류 리포트 (임계값 최적화 적용):")
print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
acc = accuracy_score(y_test, y_pred)

# 경보 발령 적중률 (관심+경계 → binary)
y_test_bin = (y_test >= 1).astype(int)
y_pred_bin = (y_pred >= 1).astype(int)
alert_precision = precision_score(y_test_bin, y_pred_bin, zero_division=0)
alert_recall    = recall_score(y_test_bin, y_pred_bin, zero_division=0)
alert_f1        = f1_score(y_test_bin, y_pred_bin, zero_division=0)

metrics = {
    "lead_days": LEAD_DAYS,
    "test_period": "2024-01-01 ~ 2025-12-22",
    "accuracy": round(acc, 4),
    "macro_f1": round(macro_f1, 4),
    "weighted_f1": round(weighted_f1, 4),
    "alert_precision": round(alert_precision, 4),
    "alert_recall": round(alert_recall, 4),
    "alert_f1": round(alert_f1, 4),
}
print(f"\n  Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}")
print(f"  경보 Precision: {alert_precision:.4f}  Recall: {alert_recall:.4f}  F1: {alert_f1:.4f}")

# ── 7b. 연도별 성능 분석 ──────────────────────────────────────────────────
print("\n[6b] 연도별 성능 분석")

df_test_tmp = df.loc[test_mask].copy()
df_test_tmp["pred_stage_tmp"] = y_pred
yearly_rows = []
for yr in sorted(df_test_tmp["조사일"].dt.year.unique()):
    mask_yr = df_test_tmp["조사일"].dt.year == yr
    yt = df_test_tmp.loc[mask_yr, "target_d7"].values
    yp = df_test_tmp.loc[mask_yr, "pred_stage_tmp"].values
    if len(yt) == 0:
        continue
    yr_acc   = accuracy_score(yt, yp)
    yr_mf1   = f1_score(yt, yp, average="macro", zero_division=0)
    yr_alert_r = recall_score((yt >= 1).astype(int), (yp >= 1).astype(int), zero_division=0)
    # 경계 recall
    mask_b = yt == 2
    yr_b_rec = recall_score(yt[mask_b], yp[mask_b], labels=[2], average="micro", zero_division=0) if mask_b.sum() > 0 else float("nan")
    yearly_rows.append({
        "year": yr, "n": int(len(yt)),
        "accuracy": round(yr_acc, 4),
        "macro_f1": round(yr_mf1, 4),
        "alert_recall": round(yr_alert_r, 4),
        "alert_boundary_recall": round(yr_b_rec, 4) if not np.isnan(yr_b_rec) else None,
    })
    print(f"  {yr}: n={len(yt):,}  acc={yr_acc:.3f}  macro-F1={yr_mf1:.3f}  경계Recall={yr_b_rec:.3f}")

yearly_df = pd.DataFrame(yearly_rows)
yearly_df.to_csv(OUT / "reports" / "metrics_by_year.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, title in zip(axes, ["accuracy", "macro_f1", "alert_boundary_recall"],
                           ["Accuracy", "Macro-F1", "경계 Recall"]):
    ax.bar(yearly_df["year"].astype(str), yearly_df[col], color="#1565C0", alpha=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_xlabel("연도")
plt.suptitle("연도별 모델 성능 (테스트 세트)", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "metrics_by_year.png", dpi=150)
plt.close()
print("  저장: metrics_by_year.png")

# ── 8. 혼동행렬 저장 ────────────────────────────────────────────────────────
print("\n[7] 혼동행렬 저장")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(OUT / "reports" / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title(f"혼동행렬 (7일 선행 예측, 테스트 2024~)", fontsize=13)
ax.set_ylabel("실제")
ax.set_xlabel("예측")
plt.tight_layout()
fig.savefig(OUT / "plots" / "confusion_matrix.png", dpi=150)
plt.close()

# ── 9. 예측 시계열 그래프 ───────────────────────────────────────────────────
print("[8] 예측 시계열 그래프 저장")

df_test = df.loc[test_mask].copy()
df_test["pred_stage"] = y_pred
df_test["prob_normal"] = y_pred_prob[:, 0]
df_test["prob_caution"] = y_pred_prob[:, 1]
df_test["prob_alert"]   = y_pred_prob[:, 2]
df_test.to_csv(OUT / "reports" / "holdout_predictions.csv", index=False, encoding="utf-8-sig")

for site in df_test["채수위치"].unique():
    sub = df_test[df_test["채수위치"] == site].sort_values("조사일")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 남조류 농도
    axes[0].plot(sub["조사일"], sub["total_cyano"], color="#2196F3", lw=1.2, label="total_cyano")
    axes[0].axhline(1000, color="orange", ls="--", lw=1, label="관심 기준(1,000)")
    axes[0].axhline(10000, color="red", ls="--", lw=1, label="경계 기준(10,000)")
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_ylabel("유해남조류 (cells/mL)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{site} – 유해남조류 농도", fontsize=11)

    # 실제 vs 예측 단계
    stage_colors = {0: "green", 1: "orange", 2: "red"}
    for _, row in sub.iterrows():
        axes[1].axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["target_d7"]), "gray"), alpha=0.4)
        axes[1].axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["pred_stage"]), "gray"), alpha=0.15)
    from matplotlib.patches import Patch
    legend_els = [Patch(color="green", alpha=0.4, label="미발령"),
                  Patch(color="orange", alpha=0.4, label="관심"),
                  Patch(color="red", alpha=0.4, label="경계")]
    axes[1].set_yticks([])
    axes[1].legend(handles=legend_els, loc="upper right", fontsize=8)
    axes[1].set_title("발령단계: 실제(진함) vs 예측(연함)", fontsize=11)

    # 예측 확률
    axes[2].stackplot(sub["조사일"],
                      sub["prob_normal"], sub["prob_caution"], sub["prob_alert"],
                      labels=["미발령", "관심", "경계"],
                      colors=["#66BB6A", "#FFA726", "#EF5350"], alpha=0.8)
    axes[2].set_ylabel("예측 확률")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("7일 선행 경보 발령 확률", fontsize=11)

    plt.suptitle(f"대청댐 {site} – 7일 선행 남조류 경보 예측 (테스트: 2024~)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "plots" / f"timeseries_{site}.png", dpi=150)
    plt.close()
    print(f"  저장: timeseries_{site}.png")

# ── 10. SHAP 분석 ────────────────────────────────────────────────────────────
print("\n[9] SHAP 분석 – 주요 영향인자 도출")

explainer = shap.TreeExplainer(model)
shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
shap_values_raw = explainer.shap_values(shap_sample)

# shap_values_raw: 3-D array (n_samples, n_features, n_classes) OR list of 2-D arrays
if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    # shape: (n_samples, n_features, n_classes)
    shap_values = [shap_values_raw[:, :, c] for c in range(shap_values_raw.shape[2])]
elif isinstance(shap_values_raw, list):
    shap_values = shap_values_raw
else:
    shap_values = [shap_values_raw]

# 클래스별 평균 절대값 SHAP 합산 (전체 중요도)
mean_shap = np.zeros(len(FEATURE_COLS))
for cls_shap in shap_values:
    mean_shap += np.abs(cls_shap).mean(axis=0)

shap_df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_shap})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_df.to_csv(OUT / "reports" / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")

# Top-20 피처 바 차트
top20 = shap_df.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="#1565C0")
ax.set_xlabel("평균 |SHAP 값| (전체 클래스 합산)")
ax.set_title("녹조 경보 발령 주요 영향인자 Top-20 (SHAP)", fontsize=13)
plt.tight_layout()
fig.savefig(OUT / "plots" / "shap_top20_bar.png", dpi=150)
plt.close()
print("  저장: shap_top20_bar.png")

# 경계 클래스(cls=2) beeswarm
print("  경계 클래스 SHAP beeswarm 저장 중...")
top15_idx = [FEATURE_COLS.index(f) for f in shap_df["feature"].head(15).tolist()]
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values[2][:, top15_idx],
    shap_sample.iloc[:, top15_idx],
    feature_names=[FEATURE_COLS[i] for i in top15_idx],
    plot_type="dot",
    show=False,
    max_display=15,
)
plt.title("경계 발령 SHAP Beeswarm (Top-15 피처)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "plots" / "shap_beeswarm_alert.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: shap_beeswarm_alert.png")

# ── 11. 시나리오 기반 의사결정 지원 ────────────────────────────────────────
print("\n[10] 시나리오 기반 의사결정 지원체계")

SCENARIO_RULES = {
    "S1_고온건조": {
        "desc": "기온 ≥28°C & 일사량 ≥18 MJ/m2 & 강수 ≤2mm (7일 누적)",
        "condition": lambda r: (
            r.get("평균기온(°C)_roll7m", 0) >= 28 and
            r.get("합계 일사량(MJ/m2)_roll7m", 0) >= 18 and
            r.get("일강수량(mm)_roll7sum", 99) <= 2
        ),
        "actions": [
            "조류 모니터링 주 2회 이상 강화",
            "취수구 심층수 전환 검토",
            "조류제거선 사전 대기",
        ],
    },
    "S2_저수위_체류": {
        "desc": "저수율 ≤ 50% & 유입량 감소 & 수온 ≥25°C",
        "condition": lambda r: (
            r.get("저수율(%)_lag1", 100) <= 50 and
            r.get("수온(℃)_lag1", 0) >= 25
        ),
        "actions": [
            "방류량 증가로 체류시간 단축",
            "상류 지류 수질 모니터링",
            "녹조방지 포기장치 가동",
        ],
    },
    "S3_관심단계_진입": {
        "desc": "관심 발령 확률 ≥ 0.4 또는 직전 stage=관심",
        "condition": lambda r: (
            r.get("prob_caution", 0) >= 0.4 or r.get("stage_lag1", 0) >= 1
        ),
        "actions": [
            "상수도 조류독소(MC-LR) 검사 실시",
            "정수처리 PAC 투입량 증가",
            "민원 선제적 공지 준비",
        ],
    },
    "S4_경계단계_임박": {
        "desc": "경계 발령 확률 ≥ 0.3 또는 유해남조류 ≥ 5,000 cells/mL",
        "condition": lambda r: (
            r.get("prob_alert", 0) >= 0.3 or
            r.get("total_cyano_lag1", 0) >= 5000
        ),
        "actions": [
            "경계 경보 발령 준비 및 관계기관 사전 통보",
            "취수 중단 시나리오 검토",
            "황토 살포 등 긴급 제거 조치",
            "유역 내 비점오염원 점검 강화",
        ],
    },
    "S5_강우_후_유입": {
        "desc": "7일 누적 강수 ≥ 30mm 후 고온 전환",
        "condition": lambda r: (
            r.get("일강수량(mm)_roll7sum", 0) >= 30 and
            r.get("평균기온(°C)_lag1", 0) >= 25
        ),
        "actions": [
            "강우 후 비점오염 유입 모니터링",
            "댐 유입구 탁도/T-N, T-P 측정",
            "향후 7~14일 남조류 급증 위험 주의보 발령",
        ],
    },
}

# 테스트 데이터에 시나리오 적용
scenario_records = []
for _, row in df_test.iterrows():
    r = row.to_dict()
    triggered = []
    for sid, sc in SCENARIO_RULES.items():
        try:
            if sc["condition"](r):
                triggered.append(sid)
        except Exception:
            pass
    scenario_records.append({
        "조사일": row["조사일"],
        "채수위치": row["채수위치"],
        "실제단계": labels[int(row["target_d7"])],
        "예측단계": labels[int(row["pred_stage"])],
        "prob_정상": round(row["prob_normal"], 3),
        "prob_관심": round(row["prob_caution"], 3),
        "prob_경계": round(row["prob_alert"], 3),
        "발동_시나리오": "|".join(triggered) if triggered else "없음",
        "권고조치": " / ".join(
            a for sid in triggered for a in SCENARIO_RULES[sid]["actions"][:2]
        ) if triggered else "정상 모니터링 유지",
    })

sc_df = pd.DataFrame(scenario_records)
sc_df.to_csv(OUT / "reports" / "scenario_recommendations.csv", index=False, encoding="utf-8-sig")
print(f"  시나리오 권고 레코드: {len(sc_df):,}건 저장")
print(f"  시나리오 발동 건수:\n{sc_df['발동_시나리오'].value_counts().head(10)}")

# ── 12. 메트릭/중요도 리포트 저장 ───────────────────────────────────────────
print("\n[11] 최종 리포트 저장")

report = {
    "model_info": {
        "algorithm": "LightGBM MultiClass",
        "lead_days": LEAD_DAYS,
        "n_features": len(FEATURE_COLS),
        "best_iteration": int(model.best_iteration_),
        "classes": labels,
    },
    "metrics": metrics,
    "top10_features": shap_df.head(10)[["feature", "mean_abs_shap"]].to_dict(orient="records"),
    "scenario_definitions": {
        sid: {"desc": sc["desc"], "actions": sc["actions"]}
        for sid, sc in SCENARIO_RULES.items()
    },
    "data_sources": [
        "환경부 물환경정보시스템 (water.nier.go.kr): 수질/조류 현장측정 데이터",
        "기상청 기상자료개방포털 (data.kma.go.kr): 일별 기온·강수·일사량·풍속·습도",
        "K-water 대청댐 운영현황: 수위·저수량·유입량·방류량",
        "기후에너지환경부 조류경보제 운영지침: 발령 기준 및 단계 정의",
    ],
}

with open(OUT / "reports" / "model_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# 추론용 메타데이터 (모델 배포 시 필요한 임계값·피처 목록)
inference_meta = {
    "feature_cols": FEATURE_COLS,
    "lead_days": LEAD_DAYS,
    "threshold_alert": float(best_thr_alert),
    "threshold_caution": float(best_thr_caution),
    "classes": labels,
    "stage_map": {"미발령": 0, "관심": 1, "경계": 2},
    "alert_weight_boost": ALERT_WEIGHT_BOOST,
}
with open(OUT / "models" / "inference_meta.json", "w", encoding="utf-8") as f:
    json.dump(inference_meta, f, ensure_ascii=False, indent=2)
print("  추론 메타데이터 저장: models/inference_meta.json")

# 피처 중요도 바 차트 (모델 내장)
lgb_imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(lgb_imp["feature"][::-1], lgb_imp["importance"][::-1], color="#00897B")
ax.set_xlabel("LightGBM Feature Importance (gain)")
ax.set_title("LightGBM 피처 중요도 Top-20", fontsize=13)
plt.tight_layout()
fig.savefig(OUT / "plots" / "lgb_feature_importance.png", dpi=150)
plt.close()
print("  저장: lgb_feature_importance.png")

# ── 13. Peak Timing 분석 ────────────────────────────────────────────────────
print("\n[12] Peak Timing 분석 – 경보 이벤트별 선행 예측 일수")

def find_alert_events(series_stage: pd.Series, series_date: pd.Series, min_stage: int = 2):
    """연속된 경보 구간을 이벤트 단위로 묶어 (start_date, end_date) 목록 반환."""
    events = []
    in_event, start = False, None
    for date, stage in zip(series_date, series_stage):
        if stage >= min_stage and not in_event:
            in_event, start = True, date
        elif stage < min_stage and in_event:
            events.append((start, date - pd.Timedelta(days=1)))
            in_event = False
    if in_event:
        events.append((start, series_date.iloc[-1]))
    return events

timing_rows = []
for site in df_test["채수위치"].unique():
    sub = df_test[df_test["채수위치"] == site].sort_values("조사일").reset_index(drop=True)
    events = find_alert_events(sub["target_d7"], sub["조사일"], min_stage=2)
    for ev_start, ev_end in events:
        # 이 이벤트 시작 기준으로 모델이 처음 '경계'를 예측한 날
        window = sub[sub["조사일"] <= ev_start].tail(30)  # 최대 30일 이전까지 탐색
        first_pred = window[window["pred_stage"] == 2]["조사일"]
        if len(first_pred) > 0:
            lead = (ev_start - first_pred.iloc[-1]).days
        else:
            lead = None   # 해당 이벤트 전에 경계 예측 없음 (miss)
        timing_rows.append({
            "채수위치": site,
            "이벤트_시작": ev_start.date(),
            "이벤트_종료": ev_end.date(),
            "이벤트_기간(일)": (ev_end - ev_start).days + 1,
            "예측_선행일수": lead,
            "탐지여부": "탐지" if lead is not None else "미탐지",
        })

timing_df = pd.DataFrame(timing_rows)
timing_df.to_csv(OUT / "reports" / "peak_timing_analysis.csv", index=False, encoding="utf-8-sig")

detected = timing_df[timing_df["탐지여부"] == "탐지"]["예측_선행일수"]
print(f"  경계 이벤트 총 {len(timing_df)}건 | 탐지 {len(detected)}건 | 미탐지 {len(timing_df)-len(detected)}건")
if len(detected) > 0:
    print(f"  선행 예측 일수: 평균 {detected.mean():.1f}일 | 중앙값 {detected.median():.0f}일 | 최대 {detected.max():.0f}일")

if len(detected) > 0:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(detected, bins=range(0, int(detected.max()) + 2), color="#1565C0", edgecolor="white", alpha=0.85)
    ax.axvline(7, color="red", ls="--", lw=1.5, label="목표 리드타임 7일")
    ax.set_xlabel("선행 예측 일수 (경보 이벤트 시작 기준)")
    ax.set_ylabel("이벤트 수")
    ax.set_title("경계 경보 이벤트별 선행 예측 일수 분포", fontsize=12)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "plots" / "peak_timing_histogram.png", dpi=150)
    plt.close()
    print("  저장: peak_timing_histogram.png")

# ── 14. 리드타임 비교 (1 / 3 / 7 / 14일) ────────────────────────────────────
print("\n[13] 리드타임 비교 분석 (1 / 3 / 7 / 14일)")

def train_leadtime_model(df_feat: pd.DataFrame, feature_cols: list, lead: int) -> dict:
    col_name = f"_target_L{lead}"
    parts2 = []
    for site, grp in df_feat.groupby("채수위치"):
        g2 = grp.copy().sort_values("조사일")
        g2[col_name] = g2["stage_num"].shift(-lead)
        parts2.append(g2)
    df2 = pd.concat(parts2).sort_values(["조사일", "채수위치"]).dropna(subset=[col_name])
    df2[col_name] = df2[col_name].astype(int).clip(upper=2)

    # 날짜 기반 분리 (인덱스 불일치 방지)
    tr = df2[df2["조사일"] < "2023-01-01"]
    vl = df2[(df2["조사일"] >= "2023-01-01") & (df2["조사일"] < "2024-01-01")]
    te = df2[df2["조사일"] >= "2024-01-01"]

    Xtr, ytr = tr[feature_cols].dropna(), tr.loc[tr[feature_cols].dropna().index, col_name]
    Xv,  yv  = vl[feature_cols].dropna(), vl.loc[vl[feature_cols].dropna().index, col_name]
    Xte, yte = te[feature_cols].dropna(), te.loc[te[feature_cols].dropna().index, col_name]
    if len(Xtr) == 0 or len(Xte) == 0:
        return {}

    cnt = ytr.value_counts().sort_index()
    bw  = {k: len(ytr) / (len(cnt) * v) for k, v in cnt.items()}
    bw[2] = bw.get(2, 1.0) * 3.0
    sw = ytr.map(bw).values

    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, metric="multi_logloss",
        n_estimators=600, learning_rate=0.05, max_depth=6, num_leaves=63,
        min_child_samples=10, feature_fraction=0.75, bagging_fraction=0.75,
        bagging_freq=5, verbose=-1, random_state=42, n_jobs=-1,
    )
    m.fit(Xtr, ytr, sample_weight=sw,
          eval_set=[(Xv, yv)],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])

    prob = m.predict_proba(Xte)
    yp = np.zeros(len(yte), dtype=int)
    yp[prob[:, 2] >= 0.12] = 2
    yp[(prob[:, 1] >= 0.30) & (yp < 2)] = 1

    b_rec  = recall_score((yte >= 1).astype(int), (yp >= 1).astype(int), zero_division=0)
    b_pre  = precision_score((yte >= 1).astype(int), (yp >= 1).astype(int), zero_division=0)
    mask2  = (yte == 2).values
    c_rec  = recall_score(yte.values[mask2], yp[mask2], labels=[2], average="micro", zero_division=0) if mask2.sum() > 0 else float("nan")
    mf1    = f1_score(yte, yp, average="macro", zero_division=0)
    return {
        "lead_days": lead,
        "n_test": int(len(yte)),
        "macro_f1": round(float(mf1), 4),
        "alert_recall": round(float(b_rec), 4),
        "alert_precision": round(float(b_pre), 4),
        "boundary_recall": round(float(c_rec), 4) if not np.isnan(c_rec) else None,
    }

lead_results = []
for lead_d in [1, 3, 7, 14]:
    print(f"  리드타임 {lead_d:2d}일 학습 중...", end=" ", flush=True)
    res = train_leadtime_model(df, FEATURE_COLS, lead_d)
    if res:
        lead_results.append(res)
        print(f"macro-F1={res['macro_f1']:.3f}  경보Recall={res['alert_recall']:.3f}  경계Recall={res.get('boundary_recall', 'N/A')}")

lead_df = pd.DataFrame(lead_results)
lead_df.to_csv(OUT / "reports" / "leadtime_comparison.csv", index=False, encoding="utf-8-sig")

if len(lead_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title, color in zip(
        axes,
        ["macro_f1", "alert_recall", "boundary_recall"],
        ["Macro-F1", "경보 Recall (관심+경계)", "경계 Recall"],
        ["#1565C0", "#2E7D32", "#C62828"],
    ):
        vals = lead_df[col].astype(float)
        ax.plot(lead_df["lead_days"], vals, "o-", color=color, lw=2, ms=8)
        ax.set_xlabel("리드타임 (일)")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([1, 3, 7, 14])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.4)
    plt.suptitle("리드타임별 모델 성능 비교 (1 / 3 / 7 / 14일)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "plots" / "leadtime_comparison.png", dpi=150)
    plt.close()
    print("  저장: leadtime_comparison.png")

# ── 13. 완료 요약 ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
print(f"  [모델 성능] Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
print(f"  [경보 적중] Precision={alert_precision:.4f}  Recall={alert_recall:.4f}  F1={alert_f1:.4f}")
print(f"\n  [출력 파일]")
print(f"    outputs/plots/   : {list((OUT/'plots').glob('*.png'))}")
print(f"    outputs/reports/ : {list((OUT/'reports').glob('*'))}")
print()
print("  [주요 영향 인자 Top-5]")
for _, row in shap_df.head(5).iterrows():
    print(f"    {row['feature']:<45} SHAP={row['mean_abs_shap']:.4f}")
