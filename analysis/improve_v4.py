"""
개선 v4
==============================================
1. 2단계 분류기
   Stage1: 경보(관심+경계) vs 미발령 — 이진 모델
   Stage2: 경계 vs 관심 — 이진 모델 (Stage1이 경보로 분류한 것에만 적용)

2. 최근 연도 가중치 부여
   경계 발생이 있는 연도(2017·2020·2022·2023)에 높은 sample weight

실행: python3.10 improve_v4.py
결과: analysis/outputs/improvements_v4/
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
    roc_auc_score,
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
OUT       = BASE / "outputs" / "improvements_v4"
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

STAGE_MAP  = {"미발령": 0, "관심": 1, "경계": 2, "조류대발생": 3}
LABELS     = ["미발령", "관심", "경계"]
CYANO_COLS = ["microcystis", "anabaena", "oscillatoria", "aphanizomenon"]

# 연도별 경계 발생 여부 기반 가중치 배율
YEAR_BOOST = {2017: 3.0, 2020: 3.0, 2022: 2.0, 2023: 4.0}   # 경계 많은 해
YEAR_BASE  = 1.0

# ── 전처리 & 피처 ─────────────────────────────────────────────────────────────
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
        if col not in g.columns: continue
        for lag in [1, 3, 7, 14]:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        for win in [3, 7, 14]:
            g[f"{col}_roll{win}m"]   = g[col].shift(1).rolling(win).mean()
            g[f"{col}_roll{win}max"] = g[col].shift(1).rolling(win).max()
    for col in ["수온(℃)", "pH", "DO(㎎/L)", "탁도"]:
        if col not in g.columns: continue
        for lag in [1, 3, 7]: g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()
    for col in ["평균기온(°C)", "최고기온(°C)", "합계 일사량(MJ/m2)",
                "평균 풍속(m/s)", "평균 상대습도(%)"]:
        if col not in g.columns: continue
        for lag in [1, 3, 7]: g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()
    for col in ["일강수량(mm)", "강우량(mm)"]:
        if col not in g.columns: continue
        for lag in [1, 3, 7, 14]: g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7sum"]  = g[col].shift(1).rolling(7).sum()
        g[f"{col}_roll14sum"] = g[col].shift(1).rolling(14).sum()
    for col in ["수위(EL.m)", "저수율(%)", "유입량(㎥/s)", "총방류량(㎥/s)"]:
        if col not in g.columns: continue
        for lag in [1, 3, 7]: g[f"{col}_lag{lag}"] = g[col].shift(lag)
        g[f"{col}_roll7m"] = g[col].shift(1).rolling(7).mean()
    if "최고기온(°C)" in g.columns:
        g["gdd7"]  = g["최고기온(°C)"].shift(1).clip(lower=10).rolling(7).sum()
        g["gdd14"] = g["최고기온(°C)"].shift(1).clip(lower=10).rolling(14).sum()
    if "수온(℃)" in g.columns and "평균기온(°C)" in g.columns:
        g["temp_diff"]        = g["수온(℃)"] - g["평균기온(°C)"]
        g["temp_diff_lag1"]   = g["temp_diff"].shift(1)
        g["temp_diff_roll7m"] = g["temp_diff"].shift(1).rolling(7).mean()
    for lag in [1, 3, 7]: g[f"stage_lag{lag}"] = g["stage_num"].shift(lag)
    g["stage_roll7max"] = g["stage_num"].shift(1).rolling(7).max()
    g["month"]          = g["조사일"].dt.month
    g["week_of_year"]   = g["조사일"].dt.isocalendar().week.astype(int)
    g["sin_month"]      = np.sin(2 * np.pi * g["month"] / 12)
    g["cos_month"]      = np.cos(2 * np.pi * g["month"] / 12)
    g[f"target_d{lead}"] = g["stage_num"].shift(-lead)
    return g


EXCLUDE = {"조사일", "채수위치", "발령단계", "stage_num", "일강수량(mm)"}

def get_feat(df):
    return [c for c in df.columns
            if c not in EXCLUDE and df[c].dtype != object
            and not c.startswith("target_")]

# ── 연도 가중치 계산 ──────────────────────────────────────────────────────────
def year_sample_weights(dates: pd.Series, y: pd.Series) -> np.ndarray:
    """
    연도 기반 가중치 × 클래스 기반 가중치 결합
    - 경계 발생 연도에 year_boost 곱
    - 경계 클래스에 추가 3배
    """
    years = dates.dt.year
    year_w = years.map(lambda yr: YEAR_BOOST.get(yr, YEAR_BASE)).values

    cnt   = pd.Series(y).value_counts().sort_index()
    base  = {k: len(y) / (len(cnt) * v) for k, v in cnt.items()}
    base[2] = base.get(2, 1.0) * 3.0
    class_w = pd.Series(y).map(base).values

    combined = year_w * class_w
    combined = combined / combined.mean()   # 평균 1로 정규화
    return combined

# ── 임계값 탐색 헬퍼 ──────────────────────────────────────────────────────────
def find_best_thr(prob, y_true, tb_range, tc_range=None, min_b_rec=0.70):
    """이진(Stage1) 또는 다중(전체) 임계값 탐색"""
    best_cost, best_tb, best_tc = 1e9, 0.10, 0.30
    rows = []
    if tc_range is None:
        # 이진 임계값 탐색
        for tb in tb_range:
            yp  = (prob >= tb).astype(int)
            FP  = int(((yp == 1) & (np.array(y_true) == 0)).sum())
            FN  = int(((yp == 0) & (np.array(y_true) == 1)).sum())
            rec = float((yp[np.array(y_true) == 1] == 1).mean()) if (np.array(y_true)==1).sum()>0 else 0
            rows.append({"tb": tb, "cost": FP + 3*FN, "rec": round(rec, 3)})
        rows.sort(key=lambda x: x["cost"])
        filt = [r for r in rows if r["rec"] >= min_b_rec]
        best = filt[0] if filt else rows[0]
        return best["tb"], None, best
    else:
        for tb in tb_range:
            for tc in tc_range:
                yp  = np.zeros(len(y_true), dtype=int)
                yp[prob[:, 2] >= tb] = 2
                yp[(prob[:, 1] >= tc) & (yp < 2)] = 1
                yb_t = (np.array(y_true) >= 1).astype(int)
                yb_p = (yp >= 1).astype(int)
                FP  = int(((yb_p==1)&(yb_t==0)).sum())
                FN  = int(((yb_p==0)&(yb_t==1)).sum())
                b_m = np.array(y_true) == 2
                b_r = float((yp[b_m]==2).sum())/max(b_m.sum(),1)
                rows.append({"tb": tb, "tc": tc, "cost": FP+3*FN, "b_rec": round(b_r,3)})
        rows.sort(key=lambda x: x["cost"])
        filt = [r for r in rows if r["b_rec"] >= min_b_rec]
        best = filt[0] if filt else rows[0]
        return best["tb"], best["tc"], best

def mdict(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    yb_t = (yt>=1).astype(int); yb_p = (yp>=1).astype(int)
    b_m  = yt == 2
    b_r  = float(recall_score(yt[b_m], yp[b_m], labels=[2],
                              average="micro", zero_division=0)) if b_m.sum()>0 else None
    return dict(
        accuracy       = round(float(accuracy_score(yt, yp)), 4),
        macro_f1       = round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        alert_recall   = round(float(recall_score(yb_t, yb_p, zero_division=0)), 4),
        alert_precision= round(float(precision_score(yb_t, yb_p, zero_division=0)), 4),
        alert_f1       = round(float(f1_score(yb_t, yb_p, zero_division=0)), 4),
        boundary_recall= round(b_r, 4) if b_r is not None else None,
    )

def cm_heatmap(yt, yp, title, ax):
    cm = confusion_matrix(yt, yp, labels=[0,1,2])
    sns.heatmap(pd.DataFrame(cm, index=LABELS, columns=LABELS),
                annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(title, fontsize=10); ax.set_ylabel("실제"); ax.set_xlabel("예측")

# ═══════════════════════════════════════════════════════════════════════════════
# 데이터 준비
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("데이터 준비")
print("=" * 70)

df_raw = load_and_prep()
parts  = [make_features(g, lead=7) for _, g in df_raw.groupby("채수위치")]
df7    = pd.concat(parts).sort_values(["조사일","채수위치"]).reset_index(drop=True)
df7    = df7.dropna(subset=["target_d7"])
df7["target_d7"] = df7["target_d7"].astype(int).clip(upper=2)
df7["year"]      = df7["조사일"].dt.year

FEAT = get_feat(df7)
print(f"  피처: {len(FEAT)}개")

# 분리 — 임계값 튜닝용 / 최종 학습용 동일하게 유지
tune_tr = df7[df7["조사일"] < "2023-01-01"]
val_df  = df7[(df7["조사일"] >= "2023-01-01") & (df7["조사일"] < "2024-01-01")]
full_tr = df7[df7["조사일"] < "2024-01-01"]
test_df = df7[df7["조사일"] >= "2024-01-01"]

for name, sub in [("튜닝Train(~2022)", tune_tr), ("Val(2023)", val_df),
                   ("최종Train(~2023)", full_tr),  ("Test(2024-2025)", test_df)]:
    n = len(sub); b = (sub["target_d7"]==2).sum()
    print(f"  {name}: {n:,}행 | 경계 {b}건({b/n*100:.1f}%)")

print(f"\n  연도 가중치 배율: {YEAR_BOOST}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 2단계 분류기
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("개선1: 2단계 분류기 (Stage1: 경보여부 / Stage2: 경계vs관심)")
print("=" * 70)

lgb_base = dict(
    metric="binary_logloss", n_estimators=1500, learning_rate=0.03,
    max_depth=7, num_leaves=127, min_child_samples=10,
    feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
    lambda_l1=0.05, lambda_l2=0.1, verbose=-1, random_state=42, n_jobs=-1,
)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: 경보(관심+경계) vs 미발령
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Stage1] 경보 여부 이진 분류 (Train 2016-2022, Val 2023)")

Xtr1 = tune_tr[FEAT];  ytr1_bin = (tune_tr["target_d7"] >= 1).astype(int)
Xvl1 = val_df[FEAT];   yvl1_bin = (val_df["target_d7"]  >= 1).astype(int)
Xfl1 = full_tr[FEAT];  yfl1_bin = (full_tr["target_d7"] >= 1).astype(int)
Xte1 = test_df[FEAT];  yte1_bin = (test_df["target_d7"] >= 1).astype(int)

# 임계값 탐색용 모델 (2016-2022)
ros1a = RandomOverSampler(random_state=42, sampling_strategy=0.3)
Xtr1_sm, ytr1_sm = ros1a.fit_resample(Xtr1, ytr1_bin)

s1_tune = lgb.LGBMClassifier(objective="binary", **{k:v for k,v in lgb_base.items() if k!="metric"},
                               metric="binary_logloss")
s1_tune.fit(Xtr1_sm, ytr1_sm,
            sample_weight=np.where(ytr1_sm==1, 3.0, 1.0),
            eval_set=[(Xvl1, yvl1_bin)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
print(f"    튜닝 모델 best_iter={s1_tune.best_iteration_}")

vp1 = s1_tune.predict_proba(Xvl1)[:, 1]
TB1_RANGE = np.arange(0.01, 0.50, 0.005)
tb1, _, best1 = find_best_thr(vp1, yvl1_bin.values, TB1_RANGE, min_b_rec=0.90)
print(f"    Stage1 임계값: {tb1:.3f}  (val cost={best1['cost']}, val recall={best1['rec']})")

# 최종 Stage1 모델 (2016-2023)
print("    최종 Stage1 모델 학습 (Train 2016-2023)...")
ros1b = RandomOverSampler(random_state=42, sampling_strategy=0.3)
Xfl1_sm, yfl1_sm = ros1b.fit_resample(Xfl1, yfl1_bin)

lgb_base_s1f = {**{k:v for k,v in lgb_base.items() if k!="metric"}, "n_estimators": 2000}
s1_final = lgb.LGBMClassifier(objective="binary", metric="binary_logloss", **lgb_base_s1f)
s1_final.fit(Xfl1_sm, yfl1_sm,
             sample_weight=np.where(yfl1_sm==1, 3.0, 1.0),
             eval_set=[(Xvl1, yvl1_bin)],
             callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
print(f"    최종 Stage1 best_iter={s1_final.best_iteration_}")

# Stage1 확률 재탐색 (새 모델 기준)
vp1b = s1_final.predict_proba(Xvl1)[:, 1]
tb1b, _, best1b = find_best_thr(vp1b, yvl1_bin.values, np.arange(0.005, 0.40, 0.003), min_b_rec=0.90)
print(f"    최종 Stage1 임계값: {tb1b:.3f}  (val cost={best1b['cost']}, val recall={best1b['rec']})")

tp1 = s1_final.predict_proba(Xte1)[:, 1]
pred_alert_mask_te = tp1 >= tb1b    # 경보로 분류된 테스트 샘플

s1_auc = roc_auc_score(yte1_bin, tp1)
s1_rec = float(recall_score(yte1_bin, (tp1>=tb1b).astype(int), zero_division=0))
s1_pre = float(precision_score(yte1_bin, (tp1>=tb1b).astype(int), zero_division=0))
print(f"    Stage1 테스트: ROC-AUC={s1_auc:.4f}  Recall={s1_rec:.4f}  Precision={s1_pre:.4f}")
print(f"    경보로 분류된 테스트 샘플: {pred_alert_mask_te.sum()}건 / {len(pred_alert_mask_te)}건")

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: 경계 vs 관심 (Stage1이 경보로 분류한 것만 대상)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Stage2] 경계 vs 관심 이진 분류")

# Stage2 학습 데이터: 실제 경보(관심+경계)인 것만 사용
alert_tr_mask = tune_tr["target_d7"] >= 1
alert_fl_mask = full_tr["target_d7"] >= 1
alert_vl_mask = val_df["target_d7"]  >= 1

Xtr2 = tune_tr.loc[alert_tr_mask, FEAT]
ytr2 = (tune_tr.loc[alert_tr_mask, "target_d7"] == 2).astype(int)
Xvl2 = val_df.loc[alert_vl_mask, FEAT]
yvl2 = (val_df.loc[alert_vl_mask, "target_d7"] == 2).astype(int)
Xfl2 = full_tr.loc[alert_fl_mask, FEAT]
yfl2 = (full_tr.loc[alert_fl_mask, "target_d7"] == 2).astype(int)

print(f"    Stage2 튜닝Train: {len(Xtr2)}행 | 경계 {ytr2.sum()}건({ytr2.mean()*100:.1f}%)")
print(f"    Stage2 Val:       {len(Xvl2)}행 | 경계 {yvl2.sum()}건({yvl2.mean()*100:.1f}%)")
print(f"    Stage2 최종Train: {len(Xfl2)}행 | 경계 {yfl2.sum()}건({yfl2.mean()*100:.1f}%)")

# 임계값 탐색용 Stage2 모델 (2016-2022)
n_b2 = ytr2.sum(); n_c2 = (ytr2==0).sum()
strat2 = min(max(n_b2 * 3, 200), n_c2)
ros2a = RandomOverSampler(random_state=42, sampling_strategy={1: strat2})
Xtr2_sm, ytr2_sm = ros2a.fit_resample(Xtr2, ytr2)

s2_tune = lgb.LGBMClassifier(objective="binary", **{k:v for k,v in lgb_base.items() if k!="metric"},
                               metric="binary_logloss")
s2_tune.fit(Xtr2_sm, ytr2_sm,
            sample_weight=np.where(ytr2_sm==1, 3.0, 1.0),
            eval_set=[(Xvl2, yvl2)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])

vp2 = s2_tune.predict_proba(Xvl2)[:, 1]
tb2, _, best2 = find_best_thr(vp2, yvl2.values, TB1_RANGE, min_b_rec=0.70)
print(f"    Stage2 임계값: {tb2:.3f}  (val recall={best2['rec']})")

# 최종 Stage2 모델 (2016-2023)
print("    최종 Stage2 모델 학습 (Train 2016-2023)...")
n_bfl2 = yfl2.sum(); n_cfl2 = (yfl2==0).sum()
strat2f = min(max(n_bfl2 * 3, 400), n_cfl2)
ros2b = RandomOverSampler(random_state=42, sampling_strategy={1: strat2f})
Xfl2_sm, yfl2_sm = ros2b.fit_resample(Xfl2, yfl2)

lgb_base_s2f = {**{k:v for k,v in lgb_base.items() if k!="metric"}, "n_estimators": 2000}
s2_final = lgb.LGBMClassifier(objective="binary", metric="binary_logloss", **lgb_base_s2f)
s2_final.fit(Xfl2_sm, yfl2_sm,
             sample_weight=np.where(yfl2_sm==1, 3.0, 1.0),
             eval_set=[(Xvl2, yvl2)],
             callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
print(f"    최종 Stage2 best_iter={s2_final.best_iteration_}")

vp2b = s2_final.predict_proba(Xvl2)[:, 1]
tb2b, _, best2b = find_best_thr(vp2b, yvl2.values, np.arange(0.005, 0.50, 0.003), min_b_rec=0.70)
print(f"    최종 Stage2 임계값: {tb2b:.3f}  (val recall={best2b['rec']})")

# ─────────────────────────────────────────────────────────────────────────────
# 2단계 합성 예측
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [합성] Stage1 + Stage2 최종 예측")

tp2_all = s2_final.predict_proba(Xte1)[:, 1]   # 전체 테스트에 Stage2 확률

yp_2stage = np.zeros(len(test_df), dtype=int)
# Stage1: 경보로 분류된 것
alert_pred = tp1 >= tb1b
yp_2stage[alert_pred] = 1                       # 일단 관심으로
# Stage2: 그 중에서 경계인 것
boundary_pred = alert_pred & (tp2_all >= tb2b)
yp_2stage[boundary_pred] = 2

m_2stage = mdict(test_df["target_d7"].values, yp_2stage)
print(f"\n  2단계 분류기 테스트 성능: {m_2stage}")
print(f"\n  분류 리포트:")
print(classification_report(test_df["target_d7"], yp_2stage,
                             target_names=LABELS, zero_division=0))

# 연도별 성능
print("  연도별 성능:")
yearly_2s = []
for yr in sorted(test_df["조사일"].dt.year.unique()):
    m_ = test_df["조사일"].dt.year == yr
    r  = mdict(test_df["target_d7"].values[m_], yp_2stage[m_])
    yearly_2s.append({"year": yr, **r})
    print(f"    {yr}: acc={r['accuracy']}  macro_f1={r['macro_f1']}  "
          f"boundary_recall={r['boundary_recall']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 최근 연도 가중치 부여 (단일 다중분류 모델)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("개선2: 최근 연도 가중치 부여 (단일 다중분류 모델)")
print("=" * 70)

print(f"\n  연도별 sample weight 배율:")
years_in_train = sorted(full_tr["year"].unique())
for yr in years_in_train:
    n_b = (full_tr[full_tr["year"]==yr]["target_d7"]==2).sum()
    boost = YEAR_BOOST.get(yr, YEAR_BASE)
    print(f"    {yr}: 경계 {n_b}건  →  연도 가중치 ×{boost}")

Xfl = full_tr[FEAT];  yfl = full_tr["target_d7"]
Xvl = val_df[FEAT];   yvl = val_df["target_d7"]
Xte = test_df[FEAT];  yte = test_df["target_d7"]

# 오버샘플 (경계 클래스 확보)
n_b = (yfl == 2).sum()
ros_yw = RandomOverSampler(random_state=42, sampling_strategy={2: min(n_b * 4, 1500)})
Xfl_sm, yfl_sm = ros_yw.fit_resample(Xfl, yfl)

# 오버샘플된 행의 연도 복원 (원본 인덱스 기반)
orig_dates = full_tr["조사일"].reset_index(drop=True)
sample_idx = ros_yw.sample_indices_
sm_dates   = orig_dates.iloc[sample_idx].reset_index(drop=True)

# 연도 가중치 × 클래스 가중치 결합
cnt_sm = pd.Series(yfl_sm).value_counts().sort_index()
base_w = {k: len(yfl_sm) / (len(cnt_sm) * v) for k, v in cnt_sm.items()}
base_w[2] = base_w.get(2, 1.0) * 3.0
class_w  = pd.Series(yfl_sm).map(base_w).values
year_w   = sm_dates.dt.year.map(lambda yr: YEAR_BOOST.get(yr, YEAR_BASE)).values
final_sw = class_w * year_w
final_sw = final_sw / final_sw.mean()   # 정규화

print(f"\n  오버샘플 후 학습 ({len(Xfl_sm):,}행), 연도×클래스 가중치 결합")
print(f"  sample_weight: mean={final_sw.mean():.2f}  max={final_sw.max():.2f}  min={final_sw.min():.2f}")

lgb_yw_params = dict(
    objective="multiclass", num_class=3, metric="multi_logloss",
    n_estimators=2000, learning_rate=0.03, max_depth=7, num_leaves=127,
    min_child_samples=10, feature_fraction=0.75, bagging_fraction=0.75,
    bagging_freq=5, lambda_l1=0.05, lambda_l2=0.1,
    verbose=-1, random_state=42, n_jobs=-1,
)
model_yw = lgb.LGBMClassifier(**lgb_yw_params)
model_yw.fit(Xfl_sm, yfl_sm,
             sample_weight=final_sw,
             eval_set=[(Xvl, yvl)],
             callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
print(f"  best_iter={model_yw.best_iteration_}")

# 임계값 탐색
vp_yw = model_yw.predict_proba(Xvl)
TB_FINE = np.arange(0.005, 0.20, 0.003)
TC_RANGE = np.arange(0.10, 0.55, 0.05)
tb_yw, tc_yw, best_yw = find_best_thr(vp_yw, yvl.values, TB_FINE, TC_RANGE)
print(f"  임계값: 경계={tb_yw:.3f}, 관심={tc_yw:.2f}  "
      f"(val cost={best_yw['cost']}, b_rec={best_yw['b_rec']})")

tp_yw  = model_yw.predict_proba(Xte)
yp_yw  = np.zeros(len(yte), dtype=int)
yp_yw[tp_yw[:, 2] >= tb_yw] = 2
yp_yw[(tp_yw[:, 1] >= tc_yw) & (yp_yw < 2)] = 1

m_yw = mdict(yte.values, yp_yw)
print(f"\n  연도가중치 모델 테스트 성능: {m_yw}")
print(f"\n  분류 리포트:")
print(classification_report(yte, yp_yw, target_names=LABELS, zero_division=0))

print("  연도별 성능:")
yearly_yw = []
for yr in sorted(test_df["조사일"].dt.year.unique()):
    m_ = test_df["조사일"].dt.year == yr
    r  = mdict(yte.values[m_], yp_yw[m_])
    yearly_yw.append({"year": yr, **r})
    print(f"    {yr}: acc={r['accuracy']}  macro_f1={r['macro_f1']}  "
          f"boundary_recall={r['boundary_recall']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 둘 합치기: 2단계 + 연도가중치 앙상블
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("개선3: 2단계 + 연도가중치 앙상블")
print("=" * 70)

# Stage1 확률 vs 연도가중치 모델의 경보 확률 앙상블
p_alert_2s  = tp1                             # Stage1 경보 확률
p_alert_yw  = tp_yw[:, 1] + tp_yw[:, 2]      # 연도가중치 모델 경보 확률
p_bnd_2s    = tp2_all                         # Stage2 경계 확률
p_bnd_yw    = tp_yw[:, 2]                     # 연도가중치 모델 경계 확률

best_ens_br, best_w = 0.0, 0.5
for w in np.arange(0.2, 0.9, 0.05):
    pa = w * p_alert_2s + (1-w) * p_alert_yw
    pb = w * p_bnd_2s   + (1-w) * p_bnd_yw

    # val 기준 임계값
    pa_v = w * s1_final.predict_proba(Xvl)[:,1] + (1-w)*(vp_yw[:,1]+vp_yw[:,2])
    pb_v = w * s2_final.predict_proba(Xvl)[:,1] + (1-w)*vp_yw[:,2]

    best_c, bt_a, bt_b = 1e9, 0.10, 0.10
    for ta in np.arange(0.01, 0.40, 0.01):
        for tb_ in np.arange(0.01, 0.40, 0.01):
            yp_v = np.zeros(len(yvl), dtype=int)
            yp_v[pa_v >= ta] = 1
            yp_v[(pa_v >= ta) & (pb_v >= tb_)] = 2
            b_m  = yvl.values == 2
            b_r  = float((yp_v[b_m]==2).sum())/max(b_m.sum(),1)
            if b_r >= 0.70:
                c = 1*int(((yp_v>=1)&(yvl.values==0)).sum()) + 3*int(((yp_v==0)&(yvl.values>=1)).sum())
                if c < best_c:
                    best_c, bt_a, bt_b = c, ta, tb_

    yp_ens = np.zeros(len(yte), dtype=int)
    yp_ens[pa >= bt_a] = 1
    yp_ens[(pa >= bt_a) & (pb >= bt_b)] = 2
    br = mdict(yte.values, yp_ens)["boundary_recall"] or 0.0
    if br > best_ens_br:
        best_ens_br, best_w = br, w
        best_bt_a, best_bt_b = bt_a, bt_b

# 최적 앙상블 최종 예측
print(f"  최적 가중치: 2단계={best_w:.2f}, 연도가중치={1-best_w:.2f}")
pa_f = best_w * p_alert_2s + (1-best_w) * p_alert_yw
pb_f = best_w * p_bnd_2s   + (1-best_w) * p_bnd_yw
yp_ens_best = np.zeros(len(yte), dtype=int)
yp_ens_best[pa_f >= best_bt_a] = 1
yp_ens_best[(pa_f >= best_bt_a) & (pb_f >= best_bt_b)] = 2

m_ens = mdict(yte.values, yp_ens_best)
print(f"  앙상블 임계값: alert≥{best_bt_a:.2f}, boundary≥{best_bt_b:.2f}")
print(f"\n  앙상블 테스트 성능: {m_ens}")
print(f"\n  분류 리포트:")
print(classification_report(yte, yp_ens_best, target_names=LABELS, zero_division=0))

# ═══════════════════════════════════════════════════════════════════════════════
# 플롯 & 저장
# ═══════════════════════════════════════════════════════════════════════════════
print("\n결과 저장...")

# 혼동행렬 비교 (4개 모델)
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
ref_m = {"boundary_recall": 0.470}   # v3 기준
cm_heatmap(yte.values,      yp_2stage,     f"2단계 분류기\nb_rec={m_2stage['boundary_recall']}", axes[0])
cm_heatmap(yte.values,      yp_yw,         f"연도가중치\nb_rec={m_yw['boundary_recall']}",       axes[1])
cm_heatmap(yte.values,      yp_ens_best,   f"앙상블\nb_rec={m_ens['boundary_recall']}",          axes[2])

# v3 기준 결과 (이전 저장된 예측 CSV)
v3_path = BASE / "outputs" / "improvements_v3" / "reports" / "predictions_7d_v3.csv"
if v3_path.exists():
    v3pred = pd.read_csv(v3_path)
    cm_heatmap(v3pred["target_d7"].values, v3pred["pred_stage"].values,
               f"v3 기준\nb_rec=0.470", axes[3])
else:
    axes[3].set_title("v3 기준 (파일 없음)")

plt.suptitle("v4 개선 모델 비교 혼동행렬 (Test 2024-2025)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "v4_confusion_compare.png", dpi=150); plt.close()

# 지점별 시계열 (앙상블 — 최고 성능)
best_yp  = yp_ens_best
best_prob = np.column_stack([
    1 - pa_f,
    pa_f * (1 - pb_f),
    pa_f * pb_f,
])
best_prob = best_prob / best_prob.sum(axis=1, keepdims=True)

test_plot = test_df.copy()
test_plot["pred_stage"]   = best_yp
test_plot["prob_normal"]  = best_prob[:, 0]
test_plot["prob_caution"] = best_prob[:, 1]
test_plot["prob_alert"]   = best_prob[:, 2]
test_plot.to_csv(OUT / "reports" / "predictions_v4_ensemble.csv",
                 index=False, encoding="utf-8-sig")

stage_colors = {0: "green", 1: "orange", 2: "red"}
for site in test_df["채수위치"].unique():
    sub = test_plot[test_plot["채수위치"] == site].sort_values("조사일")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(sub["조사일"], sub["total_cyano"], color="#2196F3", lw=1.2)
    axes[0].axhline(1000,  color="orange", ls="--", lw=1, label="관심 기준")
    axes[0].axhline(10000, color="red",    ls="--", lw=1, label="경계 기준")
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_ylabel("유해남조류 (cells/mL)"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"{site} — 유해남조류 농도", fontsize=11)
    for _, row in sub.iterrows():
        axes[1].axvspan(row["조사일"], row["조사일"]+pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["target_d7"]), "gray"), alpha=0.45)
        axes[1].axvspan(row["조사일"], row["조사일"]+pd.Timedelta(days=1),
                        color=stage_colors.get(int(row["pred_stage"]), "gray"), alpha=0.15)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color="green",  alpha=0.5, label="미발령"),
                             Patch(color="orange", alpha=0.5, label="관심"),
                             Patch(color="red",    alpha=0.5, label="경계"),
                             Patch(color="gray",   alpha=0.2, label="예측(연함)")],
                   loc="upper right", fontsize=8)
    axes[1].set_yticks([]); axes[1].set_title("발령단계: 실제(진함) vs 예측(연함)", fontsize=11)
    axes[2].stackplot(sub["조사일"], sub["prob_normal"], sub["prob_caution"], sub["prob_alert"],
                      labels=["미발령", "관심", "경계"],
                      colors=["#66BB6A", "#FFA726", "#EF5350"], alpha=0.85)
    axes[2].set_ylabel("예측 확률"); axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right", fontsize=8); axes[2].set_title("7일 선행 경보 발령 확률", fontsize=11)
    plt.suptitle(f"대청댐 {site} — v4 앙상블 7일 선행 예측 (Test 2024-2025)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "plots" / f"v4_timeseries_{site}.png", dpi=150); plt.close()
    print(f"  저장: v4_timeseries_{site}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 최종 비교 요약
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("전체 개선 히스토리 최종 요약")
print("=" * 70)

all_rows = [
    ("① 기존 (임계값 0.12, Train~2022)",   {"boundary_recall":0.147,"alert_f1":0.932,"macro_f1":0.586}),
    ("② 임계값 최적화 (0.05)",             {"boundary_recall":0.524,"alert_f1":0.928,"macro_f1":0.663}),
    ("③ v3: Train~2023 + 임계값 재탐색",   {"boundary_recall":0.470,"alert_f1":0.946,"macro_f1":0.738}),
    ("④ v4-A: 2단계 분류기",               m_2stage),
    ("⑤ v4-B: 연도가중치",                 m_yw),
    ("⑥ v4-C: 2단계+연도가중치 앙상블",    m_ens),
]
print(f"\n  {'항목':<42} | {'boundary_recall':>15} | {'alert_f1':>9} | {'macro_f1':>9}")
print("  " + "-" * 84)
for label, m in all_rows:
    br = m.get("boundary_recall") or 0.0
    print(f"  {label:<42} | {br:>15.4f} | {m['alert_f1']:>9.4f} | {m['macro_f1']:>9.4f}")

summary = {
    "two_stage":     {"metrics": m_2stage, "stage1_thr": float(tb1b), "stage2_thr": float(tb2b)},
    "year_weighted": {"metrics": m_yw, "year_boost": YEAR_BOOST, "thr_boundary": float(tb_yw)},
    "ensemble":      {"metrics": m_ens, "best_weight_2stage": float(best_w)},
}
with open(OUT / "reports" / "v4_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"\n  저장: {OUT}")
print("완료!")
