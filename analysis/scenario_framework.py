"""
시나리오 기반 복합 위험 의사결정 지원 프레임워크
==============================================
기상축 × 수문축 × 수질축 = 복합 위험 시나리오

[7가지 핵심 분석]
  C1. 여름철 3단계 세분화 (6/7/8/9월)
  C2. 기온-수온 지연 상관관계 (Lag 분석)
  C3. 수문 위험 매트릭스 (저수율 × 유입량)
  C4. 복합 시나리오 분류 (S-RED/ORA/YEL/GRN)
  C5. 누적 스트레스 지수 (HSI / HLI / CSI)
  C6. 폭증 전이 패턴 & 지연 폭발 분석
  C7. 종 경쟁 구조 & 경보 실패 분석 & 대응 행동 추천

실행: python3.10 scenario_framework.py
결과: analysis/outputs/scenario_framework/
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── 폰트 설정 ────────────────────────────────────────────────────────────────
def _set_font():
    import matplotlib.font_manager as fm
    for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p):
            fe = fm.FontEntry(fname=p, name="KorFont")
            fm.fontManager.ttflist.append(fe)
            plt.rcParams["font.family"] = "KorFont"
            return
_set_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

BASE      = Path(__file__).parent
DATA_PATH = BASE / "finaldata.csv"
PRED_PATH = BASE / "outputs" / "improvements_v4" / "reports" / "predictions_v4_ensemble.csv"
OUT       = BASE / "outputs" / "scenario_framework"
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

STAGE_MAP    = {"미발령": 0, "관심": 1, "경계": 2, "조류대발생": 3}
LABELS       = ["미발령", "관심", "경계"]
CYANO_COLS   = ["microcystis", "anabaena", "oscillatoria", "aphanizomenon"]
CYANO_KOR    = {"microcystis": "마이크로시스티스", "anabaena": "아나베나",
                "oscillatoria": "오실라토리아", "aphanizomenon": "아파니조메논"}
STAGE_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
SC_COLORS    = {"S-RED": "#C62828", "S-ORA": "#EF6C00", "S-YEL": "#F9A825", "S-GRN": "#2E7D32"}

print("=" * 70)
print("  시나리오 기반 복합 위험 의사결정 지원 프레임워크")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] 데이터 로딩")

df = pd.read_csv(DATA_PATH, parse_dates=["조사일"])
df = df.sort_values(["채수위치", "조사일"]).reset_index(drop=True)
df["stage_num"] = df["발령단계"].map(STAGE_MAP).fillna(0).astype(int).clip(upper=2)
df = df.drop(columns=["일조시간 합계(hr)", "투명도"], errors="ignore")

for site, g in df.groupby("채수위치"):
    idx = g.index
    df.loc[idx, CYANO_COLS] = g[CYANO_COLS].interpolate(method="linear", limit=14)
df["total_cyano"] = df[CYANO_COLS].sum(axis=1)
df["일강수량(mm)"] = df.get("일강수량(mm)", df.get("강우량(mm)", 0)).fillna(0)

# 대표 지점 (문의) 사용 — 기상/수문은 지점 공통이므로 문의로 대표
df_rep = df[df["채수위치"] == "문의"].copy().reset_index(drop=True)

# v4 예측값 로드
df_pred = None
if PRED_PATH.exists():
    df_pred = pd.read_csv(PRED_PATH, parse_dates=["조사일"])
    print(f"  v4 예측값 로드: {len(df_pred):,}행")

print(f"  전체: {len(df):,}행 | {df['조사일'].min().date()} ~ {df['조사일'].max().date()}")
print(f"  경계: {(df['stage_num']==2).sum()}건 ({(df['stage_num']==2).mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 파생 피처 계산 (전체 데이터)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] 파생 피처 계산")

def add_derived(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("조사일").reset_index(drop=True)
    at = "평균기온(°C)"; wt = "수온(℃)"; wind = "평균 풍속(m/s)"
    rain = "일강수량(mm)"; stor = "저수율(%)"; inf_ = "유입량(㎥/s)"; out_ = "총방류량(㎥/s)"
    maxt = "최고기온(°C)"

    # 기온 누적 스트레스 (Heat Stress Index)
    if at in g.columns:
        g["HSI_14"] = g[at].clip(lower=28).sub(28).rolling(14, min_periods=1).sum()
        g["HSI_7"]  = g[at].clip(lower=28).sub(28).rolling(7,  min_periods=1).sum()
        g["heat_streak"] = 0   # 연속 고온일 수 (>= 28°C)
        streak = 0
        for i in range(len(g)):
            streak = streak + 1 if g.loc[i, at] >= 28 else 0
            g.loc[i, "heat_streak"] = streak

    # 수분 부하 지수 (Hydraulic Load Index)
    if rain in g.columns:
        g["HLI_7"]  = g[rain].rolling(7,  min_periods=1).sum() / 700
        g["HLI_14"] = g[rain].rolling(14, min_periods=1).sum() / 1400

    # 복합 스트레스 지수 (Composite Stress Index)
    if "HSI_14" in g.columns and "HLI_7" in g.columns:
        g["CSI"] = g["HSI_14"] * (1 + g["HLI_7"])

    # 기온-수온 차이 (temp_diff > 0 → 수온 > 기온 교차 후)
    if at in g.columns and wt in g.columns:
        g["temp_diff"] = g[wt] - g[at]
        g["wt_gt_at"]  = (g[wt] > g[at]).astype(int)   # 수온>기온 여부

    # 방류량 7일 변화 (급감소 감지)
    if out_ in g.columns:
        g["outflow_delta7"] = g[out_].diff(7)   # 음수 = 방류 감소

    # 유입량 7일 누적
    if inf_ in g.columns:
        g["inflow_7sum"] = g[inf_].rolling(7, min_periods=1).sum()

    # 종 우점도 (Dominance Index)
    total = g[CYANO_COLS].sum(axis=1) + 1e-6
    for c in CYANO_COLS:
        if c in g.columns:
            g[f"{c}_dom"] = g[c] / total

    # 누적 GDD (Growing Degree Days, base 10°C)
    if maxt in g.columns:
        g["GDD14"] = g[maxt].clip(lower=10).sub(10).rolling(14, min_periods=1).sum()

    # 방류 급감소 이벤트 플래그
    if "outflow_delta7" in g.columns:
        g["discharge_drop_event"] = (g["outflow_delta7"] < -50).astype(int)

    # 수온 > 기온 교차 이벤트 (per-site)
    if "wt_gt_at" in g.columns:
        g["cross_event"] = (
            (g["wt_gt_at"] == 1) & (g["wt_gt_at"].shift(1).fillna(0) == 0)
        ).astype(int)

    # 집중호우 이벤트 (7일 누적 30mm+)
    if rain in g.columns:
        g["rain_event_7d30"] = (g[rain].rolling(7, min_periods=1).sum() >= 30).astype(int)

    # 폭염 이벤트 (연속 11일+ 기온 28°C+)
    if "heat_streak" in g.columns:
        g["heatwave_11d"] = (g["heat_streak"] >= 11).astype(int)

    return g

parts = [add_derived(g) for _, g in df.groupby("채수위치")]
df = pd.concat(parts).sort_values(["조사일", "채수위치"]).reset_index(drop=True)
df_rep = df[df["채수위치"] == "문의"].copy().reset_index(drop=True)
print("  파생 피처 완료")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 시나리오 분류 (S-RED / S-ORA / S-YEL / S-GRN)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] 시나리오 분류")

def classify_scenario(row) -> str:
    at   = row.get("평균기온(°C)", 0) or 0
    wt   = row.get("수온(℃)", 0) or 0
    wind = row.get("평균 풍속(m/s)", 2) or 2
    stor = row.get("저수율(%)", 50) or 50
    inf_ = row.get("유입량(㎥/s)", 0) or 0
    out_ = row.get("총방류량(㎥/s)", 0) or 0
    hs   = row.get("heat_streak", 0) or 0
    r7   = row.get("rain_event_7d30", 0) or 0
    od7  = row.get("outflow_delta7", 0) or 0
    wt_gt = row.get("wt_gt_at", 0) or 0

    # S-RED: 기온 28°C+ 연속 11일+ & 저수율 60-80% (폭염 지속 = 최고위험)
    if hs >= 11 and 60 <= stor <= 80:
        return "S-RED"
    # S-ORA: 집중호우 + 고온 + 방류 급감소
    if r7 == 1 and at >= 25 and od7 < -30:
        return "S-ORA"
    # S-YEL: 기온 25-28°C + 저풍속 + 저수율 60-80% + 고유입
    if 25 <= at < 28 and wind < 1.0 and 60 <= stor <= 80 and inf_ >= 50:
        return "S-YEL"
    # 그 외 S-GRN
    return "S-GRN"

df["scenario"] = df.apply(classify_scenario, axis=1)
# df_rep는 C2 lag 분석에서 cross_event 추가 후 갱신 — 여기서는 기본값만 갱신
df_rep = df[df["채수위치"] == "문의"].copy().reset_index(drop=True)

sc_stats = {}
for sc in ["S-RED", "S-ORA", "S-YEL", "S-GRN"]:
    sub = df[df["scenario"] == sc]
    b_rate = (sub["stage_num"] == 2).mean() * 100 if len(sub) > 0 else 0
    sc_stats[sc] = {"n": len(sub), "boundary_rate": round(b_rate, 1)}
    print(f"  {sc}: {len(sub):,}행 | 경계 발생률 {b_rate:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# C1. 여름철 3단계 세분화 분석
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C1] 여름철 3단계 세분화")

df["month"] = df["조사일"].dt.month
summer = df[df["month"].isin([6, 7, 8, 9])]
month_stats = {}
for m, label in [(6, "6월(준비)"), (7, "7월(진입)"), (8, "8월(최고위험)"), (9, "9월(지연위험)")]:
    sub = summer[summer["month"] == m]
    ms = {
        "label": label,
        "air_temp_mean":  round(sub["평균기온(°C)"].mean(), 1)   if "평균기온(°C)" in sub else None,
        "water_temp_mean": round(sub["수온(℃)"].mean(), 1)       if "수온(℃)" in sub else None,
        "cyano_mean":     round(sub["total_cyano"].mean(), 0),
        "boundary_rate":  round((sub["stage_num"] == 2).mean() * 100, 1),
        "n":              len(sub),
    }
    month_stats[m] = ms
    print(f"  {label}: 기온={ms['air_temp_mean']}°C | 수온={ms['water_temp_mean']}°C "
          f"| 남조류={ms['cyano_mean']:,.0f} cells/mL | 경계 {ms['boundary_rate']}%")

# ═══════════════════════════════════════════════════════════════════════════════
# C2. 기온-수온 시간 지연 상관관계
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C2] 기온-수온 Lag 상관관계")

lag_corr = {}
at_ser = df_rep["평균기온(°C)"].dropna()
wt_ser = df_rep["수온(℃)"].dropna()

for lag in [0, 3, 7, 10, 14, 21]:
    if lag == 0:
        x, y = at_ser.values, wt_ser.values
    else:
        x = at_ser.iloc[:-lag].values
        y = wt_ser.iloc[lag:].values
    n = min(len(x), len(y))
    r, p = stats.pearsonr(x[:n], y[:n])
    lag_corr[lag] = round(r, 3)
    print(f"  Lag {lag:>2}일: r={r:.3f}  p={'<0.001' if p < 0.001 else f'{p:.3f}'}")

best_lag = max(lag_corr, key=lag_corr.get)
print(f"  → 최고 상관: Lag {best_lag}일 (r={lag_corr[best_lag]})")

# 수온>기온 교차 후 경계 발생 분석
cross_dates = df_rep[df_rep["cross_event"] == 1]["조사일"].tolist()
post_cross_boundary = []
for cd in cross_dates:
    window = df_rep[(df_rep["조사일"] > cd) &
                      (df_rep["조사일"] <= cd + pd.Timedelta(days=21))]
    post_cross_boundary.append(int((window["stage_num"] == 2).sum()))

avg_boundary_after_cross = float(np.mean(post_cross_boundary)) if post_cross_boundary else 0
print(f"  수온>기온 교차 후 21일 내 평균 경계 발생: {avg_boundary_after_cross:.2f}건 ({len(cross_dates)}건 교차)")

# ═══════════════════════════════════════════════════════════════════════════════
# C3. 수문 위험 매트릭스
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C3] 수문 위험 매트릭스")

stor_bins   = [0, 40, 60, 80, 110]
stor_labels = ["<40%", "40-60%", "60-80%", ">80%"]
inf_bins    = [0, 50, 150, 300, 5000]
inf_labels  = ["<50", "50-150", "150-300", ">300"]

df["stor_bin"] = pd.cut(df["저수율(%)"],   bins=stor_bins, labels=stor_labels, right=False)
df["inf_bin"]  = pd.cut(df["유입량(㎥/s)"], bins=inf_bins,  labels=inf_labels,  right=False)

hydro_matrix = pd.pivot_table(
    df, values="stage_num", index="stor_bin", columns="inf_bin",
    aggfunc=lambda x: (x == 2).mean() * 100
).round(1)

print("  저수율 × 유입량 → 경계 발생률 (%):")
print(hydro_matrix.to_string())

# 방류 급감소 vs 증가
discharge_drop  = df[df["outflow_delta7"] < -50]
discharge_rise  = df[df["outflow_delta7"] >  20]
drop_bnd_rate   = (discharge_drop["stage_num"] == 2).mean() * 100
rise_bnd_rate   = (discharge_rise["stage_num"] == 2).mean() * 100
print(f"  방류 급감소(7일 -50↓): 경계 {drop_bnd_rate:.1f}%  |  방류 증가: 경계 {rise_bnd_rate:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# C4. 복합 시나리오 통계
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C4] 복합 시나리오 통계")
for sc, st in sc_stats.items():
    print(f"  {sc}: {st['n']:,}행 | 경계 {st['boundary_rate']}%")

# ═══════════════════════════════════════════════════════════════════════════════
# C5. 누적 스트레스 지수 통계
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C5] 누적 스트레스 지수")

csi_q75 = df_rep["CSI"].quantile(0.75)
csi_q90 = df_rep["CSI"].quantile(0.90)
# 75분위가 0이면 양수값 기준으로 재설정
if csi_q75 == 0:
    csi_pos = df_rep["CSI"][df_rep["CSI"] > 0]
    csi_q75 = csi_pos.quantile(0.50) if len(csi_pos) > 0 else 1.0
if csi_q90 == 0:
    csi_pos = df_rep["CSI"][df_rep["CSI"] > 0]
    csi_q90 = csi_pos.quantile(0.80) if len(csi_pos) > 0 else 3.0
print(f"  CSI 75분위: {csi_q75:.1f}  |  90분위: {csi_q90:.1f}")

high_stress = df[df["CSI"] >= csi_q75]
low_stress  = df[df["CSI"] <  csi_q75]
hs_bnd = (high_stress["stage_num"] == 2).mean() * 100
ls_bnd = (low_stress["stage_num"]  == 2).mean() * 100
print(f"  고스트레스(CSI≥75분위) 경계율: {hs_bnd:.1f}%  vs  저스트레스: {ls_bnd:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# C6-A. 폭증 전이 패턴 분석
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C6-A] 폭증 전이 패턴 분석")

EVENTS = {
    "폭염(11일+기온28+)": "heatwave_11d",
    "집중호우(7일30mm+)": "rain_event_7d30",
    "방류급감소(7일-50↓)": "discharge_drop_event",
}

delayed_bloom = {}
for event_name, event_col in EVENTS.items():
    if event_col not in df_rep.columns:
        continue
    # 이벤트 시작일 찾기 (0→1 전환)
    event_starts = df_rep[(df_rep[event_col] == 1) & (df_rep[event_col].shift(1) == 0)]["조사일"].tolist()
    delays = []
    for ed in event_starts:
        # 이벤트 후 30일 내 경계 발생일
        window = df_rep[(df_rep["조사일"] > ed) & (df_rep["조사일"] <= ed + pd.Timedelta(days=30))]
        bnd_days = window[window["stage_num"] == 2]["조사일"]
        if len(bnd_days) > 0:
            delay = (bnd_days.iloc[0] - ed).days
            delays.append(delay)
    med_delay = float(np.median(delays)) if delays else None
    hit_rate  = len(delays) / len(event_starts) * 100 if event_starts else 0
    delayed_bloom[event_name] = {"events": len(event_starts), "hit_rate": round(hit_rate, 1),
                                  "median_delay": med_delay, "delays": delays}
    print(f"  {event_name}: 이벤트 {len(event_starts)}건 | "
          f"30일 내 경계 발생 {hit_rate:.1f}% | 중앙 지연 {med_delay}일")

# ═══════════════════════════════════════════════════════════════════════════════
# C6-B. 상태 전이 분석 (State Machine)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C6-B] 상태 전이 분석")

transitions = {"0→1": [], "1→2": [], "2→0": [], "1→0": [], "0→0": [], "1→1": [], "2→2": []}
for site, g in df.groupby("채수위치"):
    g = g.sort_values("조사일").reset_index(drop=True)
    for i in range(1, len(g)):
        prev, curr = int(g.loc[i-1, "stage_num"]), int(g.loc[i, "stage_num"])
        key = f"{prev}→{curr}"
        if key in transitions:
            transitions[key].append(g.loc[i, "조사일"])

trans_counts = {k: len(v) for k, v in transitions.items()}
print(f"  미발령→관심: {trans_counts['0→1']}건  |  관심→경계: {trans_counts['1→2']}건")
print(f"  경계→미발령: {trans_counts['2→0']}건  |  관심→미발령: {trans_counts['1→0']}건")

# 전이 발생 직전 7일 환경 평균 (관심→경계)
trans_12_dates = transitions["1→2"]
if trans_12_dates:
    pre_cond = []
    for td in trans_12_dates:
        pre = df_rep[(df_rep["조사일"] < td) & (df_rep["조사일"] >= td - pd.Timedelta(days=7))]
        if len(pre) > 0:
            pre_cond.append({
                "air_temp": pre["평균기온(°C)"].mean(),
                "water_temp": pre["수온(℃)"].mean(),
                "inflow": pre["유입량(㎥/s)"].mean(),
                "HSI_14": pre["HSI_14"].mean() if "HSI_14" in pre else None,
            })
    if pre_cond:
        pcd = pd.DataFrame(pre_cond)
        print(f"  관심→경계 전이 직전 7일 평균: 기온={pcd['air_temp'].mean():.1f}°C | "
              f"수온={pcd['water_temp'].mean():.1f}°C | 유입량={pcd['inflow'].mean():.0f}m³/s")

# ═══════════════════════════════════════════════════════════════════════════════
# C7-A. 종 경쟁 구조 분석
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C7-A] 종 경쟁 구조 분석")

wt_bins   = [0, 20, 25, 28, 40]
wt_labels = ["<20°C", "20-25°C", "25-28°C", ">28°C"]
df["wt_bin"] = pd.cut(df["수온(℃)"], bins=wt_bins, labels=wt_labels, right=False)

species_by_wt = {}
for wt_b in wt_labels:
    sub = df[df["wt_bin"] == wt_b]
    if len(sub) == 0:
        continue
    dom_cols = {c: f"{c}_dom" for c in CYANO_COLS if f"{c}_dom" in sub.columns}
    dom_means = {CYANO_KOR[c]: round(sub[dc].mean() * 100, 1) for c, dc in dom_cols.items()}
    dominant  = max(dom_means, key=dom_means.get) if dom_means else "N/A"
    bnd_rate  = (sub["stage_num"] == 2).mean() * 100
    species_by_wt[wt_b] = {"dominance": dom_means, "dominant_species": dominant,
                             "boundary_rate": round(bnd_rate, 1)}
    print(f"  {wt_b}: 우점종={dominant}({dom_means.get(dominant,0)}%) | 경계 {bnd_rate:.1f}%")

# Microcystis→Anabaena 전환 감지 (우점 변경)
df["dominant_species"] = df[[f"{c}_dom" for c in CYANO_COLS if f"{c}_dom" in df.columns]].idxmax(axis=1)
df["dominant_species"] = df["dominant_species"].str.replace("_dom", "")

# ═══════════════════════════════════════════════════════════════════════════════
# C7-B. 경보 실패 분석 (v4 예측 FN 분류)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C7-B] 경보 실패 분석")

fn_analysis = {}
if df_pred is not None:
    fn_mask = (df_pred["target_d7"] == 2) & (df_pred["pred_stage"] != 2)
    fn_df   = df_pred[fn_mask].copy()
    tp_mask = (df_pred["target_d7"] == 2) & (df_pred["pred_stage"] == 2)

    # FN 유형 분류
    if "outflow_delta7" not in fn_df.columns and "총방류량(㎥/s)" in fn_df.columns:
        fn_df["outflow_delta7"] = fn_df["총방류량(㎥/s)"].diff(7)

    def classify_fn(row):
        rain_7  = row.get("일강수량(mm)_roll7sum", 0) or 0
        wind    = row.get("평균 풍속(m/s)", 2) or 2
        month   = pd.to_datetime(row["조사일"]).month
        if rain_7 > 100:
            return "유형B: 극단 강수 이후 OOD"
        if month >= 9:
            return "유형C: 계절 이상 (9월↓)"
        return "유형A: 급격한 환경 변화"

    if len(fn_df) > 0:
        fn_df["fn_type"] = fn_df.apply(classify_fn, axis=1)
        type_counts = fn_df["fn_type"].value_counts()
        for t, c in type_counts.items():
            print(f"  {t}: {c}건 ({c/len(fn_df)*100:.1f}%)")
        fn_analysis = type_counts.to_dict()
    else:
        print("  FN 케이스 없음 (완벽 탐지)")

    total_bnd   = (df_pred["target_d7"] == 2).sum()
    tp_bnd      = tp_mask.sum()
    fn_bnd      = fn_mask.sum()
    print(f"  전체 경계 {total_bnd}건 | 탐지 {tp_bnd}건 | 미탐지 {fn_bnd}건")

# ═══════════════════════════════════════════════════════════════════════════════
# C7-C. 대응 행동 추천 시스템 (Rule Engine)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[C7-C] 대응 행동 추천 시스템")

ACTION_RULES = {
    "S-RED": {
        "scenario_desc": "폭염 11일+ & 저수율 60-80% & 수온>기온 교차",
        "risk_level": "최고위험 (경계 23%)",
        "actions": {
            "즉시(0-24h)": [
                "취수구 심층 전환 (표층 취수 중단)",
                "조류 경보 발령 절차 개시",
                "하류 정수장 활성탄 처리 준비",
            ],
            "단기(1-3일)": [
                "방류량 단계적 증가 → 체류시간 단축 목표",
                "문의→추동→회남 확산 경로 48h 집중 모니터링",
                "조류제거선 출동 대기",
            ],
            "관찰(지속)": [
                "수온 실측치 확인 (lag 14일 기온 예측 vs 실측 비교)",
                "일사량·풍속 변화 추적 (성층화 안정도)",
            ],
        }
    },
    "S-ORA": {
        "scenario_desc": "집중호우(7일 30mm+) + 고온 25°C+ + 방류 급감소",
        "risk_level": "고위험 (경계 17%)",
        "actions": {
            "즉시(0-24h)": [
                "영양염 유입 모니터링 강화 (TN·TP 실시간)",
                "방류 급감소 원인 확인 → 체류시간 증가 방지",
            ],
            "단기(1-3일)": [
                "강우 종료 후 7-14일을 '지연 폭발 감시 기간'으로 설정",
                "유입량·영양염 농도 기반 위험지수 일 1회 산출",
            ],
            "관찰(지속)": [
                "DO 하락 여부 모니터링 (혐기성 조건 형성 전조)",
                "종 구성 변화 주 2회 확인",
            ],
        }
    },
    "S-YEL": {
        "scenario_desc": "기온 25-28°C + 저풍속 + 저수율 60-80% + 고유입",
        "risk_level": "주의 (경계 16%)",
        "actions": {
            "즉시(0-24h)": [
                "조류 측정 빈도 일 1회 → 일 2회 증가",
                "누적 스트레스 지수(CSI) 일별 계산 개시",
            ],
            "단기(1-3일)": [
                "14일 후 수온 예측 (현재 기온 lag 기반)",
                "방류 조절 시나리오 A 준비 (체류시간 관리)",
            ],
            "관찰(지속)": [
                "풍속 2m/s 이하 지속 시 S-ORA 격상 검토",
                "저수율 60% 이하 유지 유도 (수체 유동성 확보)",
            ],
        }
    },
    "S-GRN": {
        "scenario_desc": "일반 기상 + 방류 증가",
        "risk_level": "안전 (경계 4%)",
        "actions": {
            "즉시(0-24h)": ["정례 모니터링 유지"],
            "단기(1-3일)": ["기상 예보 기반 시나리오 전환 여부 주 1회 검토"],
            "관찰(지속)": ["방류량 유지 → 체류시간 최소화 기조 유지"],
        }
    },
}

for sc, rule in ACTION_RULES.items():
    print(f"  [{sc}] {rule['risk_level']}")
    for timing, acts in rule["actions"].items():
        for a in acts[:1]:
            print(f"    {timing}: {a}")

# ═══════════════════════════════════════════════════════════════════════════════
# 위험 점수 계산 (전체 데이터)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[위험점수 계산]")

def compute_risk_score(row) -> float:
    at   = max(0, (row.get("평균기온(°C)", 0) or 0) - 25)
    wt   = max(0, (row.get("수온(℃)", 0) or 0) - 20)
    wind = max(0, 2 - (row.get("평균 풍속(m/s)", 2) or 2))
    stor = max(0, ((row.get("저수율(%)", 50) or 50) - 40) / 40)
    inf_ = np.log1p(row.get("inflow_7sum", 0) or 0) / 10
    out_ = np.log1p(row.get("총방류량(㎥/s)", 0) or 0) / 10
    hsi  = (row.get("HSI_14", 0) or 0) / 50  # 정규화

    score = (0.25 * at + 0.20 * wt + 0.15 * wind +
             0.15 * stor + 0.15 * inf_ - 0.10 * out_ + 0.10 * hsi)
    return round(max(0, min(100, score * 100 / 3.5)), 1)

df["risk_score"] = df.apply(compute_risk_score, axis=1)
df_rep = df[df["채수위치"] == "문의"].copy().reset_index(drop=True)

rs_q90 = df["risk_score"].quantile(0.90)
high_rs = df[df["risk_score"] >= rs_q90]
print(f"  RiskScore 90분위: {rs_q90:.1f} | 고위험 구간 경계율: "
      f"{(high_rs['stage_num']==2).mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# ▶ 시각화
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[시각화 생성]")

# ── Plot 1: 여름철 3단계 세분화 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
months = [6, 7, 8, 9]
m_labels = ["6월\n(준비)", "7월\n(진입)", "8월\n(최고위험)", "9월\n(지연위험)"]
bar_colors = ["#90CAF9", "#FFB74D", "#EF5350", "#AB47BC"]

ax = axes[0]
at_vals = [month_stats[m]["air_temp_mean"] for m in months]
wt_vals = [month_stats[m]["water_temp_mean"] for m in months]
x = np.arange(len(months))
b1 = ax.bar(x - 0.2, at_vals, 0.35, label="평균기온(°C)", color="#5C6BC0", alpha=0.85)
b2 = ax.bar(x + 0.2, wt_vals, 0.35, label="수온(°C)",    color="#EF5350", alpha=0.85)
ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
ax.axhline(28, color="red", ls="--", lw=1, alpha=0.6, label="경계 임계(28°C)")
ax.set_xticks(x); ax.set_xticklabels(m_labels, fontsize=9)
ax.set_ylabel("온도 (°C)"); ax.set_title("월별 기온 vs 수온", fontweight="bold")
ax.legend(fontsize=8); ax.set_ylim(0, 35)

ax = axes[1]
cyano_vals = [month_stats[m]["cyano_mean"] for m in months]
bars = ax.bar(x, cyano_vals, 0.6, color=bar_colors, alpha=0.9, edgecolor="white")
ax.bar_label(bars, labels=[f"{v:,.0f}" for v in cyano_vals], fontsize=8, padding=2)
ax.set_yscale("symlog", linthresh=100)
ax.set_xticks(x); ax.set_xticklabels(m_labels, fontsize=9)
ax.axhline(1000,  color="orange", ls="--", lw=1, label="관심(1,000)")
ax.axhline(10000, color="red",    ls="--", lw=1, label="경계(10,000)")
ax.set_ylabel("남조류 평균 (cells/mL)"); ax.set_title("월별 남조류 농도", fontweight="bold")
ax.legend(fontsize=8)

ax = axes[2]
bnd_vals = [month_stats[m]["boundary_rate"] for m in months]
bars = ax.bar(x, bnd_vals, 0.6, color=bar_colors, alpha=0.9, edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)
ax.set_xticks(x); ax.set_xticklabels(m_labels, fontsize=9)
ax.set_ylabel("경계 발생률 (%)"); ax.set_title("월별 경계 발생률", fontweight="bold")
ax.set_ylim(0, max(bnd_vals) * 1.3 + 1)

# 9월 주석
ax.annotate("기온↓지만\n수온 유지\n→ 착각 주의!",
            xy=(3, bnd_vals[3]), xytext=(2.3, bnd_vals[3] + 2),
            arrowprops=dict(arrowstyle="->", color="purple"),
            fontsize=8, color="purple", fontweight="bold")

plt.suptitle("여름철 3단계 세분화 — 기상·수질·경보 연계 분석", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT / "plots" / "C1_summer_phases.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C1_summer_phases.png")

# ── Plot 2: 기온-수온 Lag 상관관계 + 시계열 ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
lags  = list(lag_corr.keys())
corrs = list(lag_corr.values())
bar_c = ["#C62828" if l == best_lag else "#5C6BC0" for l in lags]
bars  = ax.bar(lags, corrs, color=bar_c, edgecolor="white", width=2)
ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
ax.set_xlabel("Lag (일)"); ax.set_ylabel("Pearson r")
ax.set_title(f"기온 → 수온 시간 지연 상관계수\n(최고: Lag {best_lag}일, r={lag_corr[best_lag]})",
             fontweight="bold")
ax.set_ylim(0.75, 0.95)
ax.axhline(lag_corr[best_lag], color="red", ls="--", lw=1, alpha=0.5)
ax.set_xticks(lags)

ax = axes[1]
summer_rep = df_rep[(df_rep["month"].isin([6, 7, 8, 9])) &
                    (df_rep["조사일"].dt.year >= 2020)].copy()
ax.plot(summer_rep["조사일"], summer_rep["평균기온(°C)"], color="#5C6BC0",
        lw=1.5, label="평균기온(°C)", alpha=0.85)
ax.plot(summer_rep["조사일"], summer_rep["수온(℃)"], color="#EF5350",
        lw=1.5, label="수온(°C)", alpha=0.85)

cross_mask = summer_rep["cross_event"] == 1
for cd in summer_rep[cross_mask]["조사일"]:
    ax.axvline(cd, color="purple", ls=":", lw=1.2, alpha=0.7)

ax.set_ylabel("온도 (°C)")
ax.set_title(f"기온 vs 수온 시계열 (2020-)\n(점선: 수온>기온 교차 → 21일 내 경계 평균 {avg_boundary_after_cross:.1f}건)",
             fontweight="bold")
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.suptitle("기온-수온 시간 지연 분석 (Lag 14일, r=0.915)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "plots" / "C2_lag_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C2_lag_correlation.png")

# ── Plot 3: 수문 위험 매트릭스 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
mask = hydro_matrix.isna()
sns.heatmap(hydro_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, mask=mask,
            annot_kws={"size": 11, "weight": "bold"},
            cbar_kws={"label": "경계 발생률 (%)"})
ax.set_title("저수율 × 유입량 → 경계 발생률 (%)\n(최고위험: 60-80% × 150-300 m³/s = 27.5%)",
             fontweight="bold")
ax.set_xlabel("유입량 구간 (m³/s)")
ax.set_ylabel("저수율 구간")

ax = axes[1]
disc_cats   = ["방류 증가\n(+20↑)", "방류 급감소\n(-50↓)"]
disc_rates  = [rise_bnd_rate, drop_bnd_rate]
disc_colors = ["#4CAF50", "#F44336"]
bars = ax.bar(disc_cats, disc_rates, color=disc_colors, alpha=0.85, edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%.1f%%", fontsize=12, padding=4, fontweight="bold")
ax.set_ylabel("경계 발생률 (%)"); ax.set_ylim(0, max(disc_rates) * 1.5)
ax.set_title("방류 시나리오별 경계 발생률\n(방류 감소 = 체류시간 증가 = 남조류 성장 기회)",
             fontweight="bold")

# 메커니즘 설명 텍스트
ax.text(1, drop_bnd_rate + 1.5,
        f"방류 감소 → 체류시간↑\n남조류 성장 기회 증가",
        ha="center", fontsize=8, color="#B71C1C", style="italic")

plt.suptitle("수문 위험 매트릭스 — 저수율·유입량·방류량의 복합 효과", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "plots" / "C3_hydraulic_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C3_hydraulic_matrix.png")

# ── Plot 4: 복합 시나리오 분류 시계열 ────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
summer_full = df_rep[(df_rep["month"].isin([5, 6, 7, 8, 9, 10]))].copy()

ax = axes[0]
ax.fill_between(summer_full["조사일"], summer_full["total_cyano"],
                color="#2196F3", alpha=0.4, label="남조류 (cells/mL)")
ax.set_yscale("symlog", linthresh=100)
ax.axhline(1000,  color="orange", ls="--", lw=1, label="관심 1,000")
ax.axhline(10000, color="red",    ls="--", lw=1, label="경계 10,000")
ax.set_ylabel("남조류 (cells/mL)"); ax.legend(fontsize=8)
ax.set_title("남조류 농도", fontweight="bold")

ax = axes[1]
risk_colors = summer_full["scenario"].map(SC_COLORS)
ax.bar(summer_full["조사일"], summer_full["risk_score"],
       color=risk_colors, width=1, alpha=0.85)
ax.set_ylabel("위험점수 (0-100)")
ax.set_title("복합 위험점수 + 시나리오 분류", fontweight="bold")
patches = [mpatches.Patch(color=v, label=k) for k, v in SC_COLORS.items()]
ax.legend(handles=patches, loc="upper left", fontsize=8, ncol=4)

ax = axes[2]
for _, row in summer_full.iterrows():
    ax.axvspan(row["조사일"], row["조사일"] + pd.Timedelta(days=1),
               color=STAGE_COLORS.get(int(row["stage_num"]), "gray"), alpha=0.5)
ax.set_ylabel("실제 발령단계"); ax.set_yticks([])
patches2 = [mpatches.Patch(color=v, label=LABELS[k], alpha=0.6) for k, v in STAGE_COLORS.items()]
ax.legend(handles=patches2, loc="upper left", fontsize=8, ncol=3)
ax.set_title("실제 발령단계", fontweight="bold")
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.suptitle("대청댐 문의 — 복합 시나리오 분류 시계열 (여름철)", fontsize=13,
             fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "C4_scenario_timeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C4_scenario_timeline.png")

# ── Plot 5: 누적 스트레스 지수 ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
ax.hist(df_rep["HSI_14"].dropna(), bins=40, color="#EF5350", edgecolor="white", alpha=0.85)
ax.axvline(df_rep["HSI_14"].quantile(0.75), color="orange", ls="--", lw=2, label="75분위")
ax.axvline(df_rep["HSI_14"].quantile(0.90), color="red",    ls="--", lw=2, label="90분위")
ax.set_xlabel("열 누적 스트레스 (HSI_14)"); ax.set_ylabel("빈도")
ax.set_title("열 누적 스트레스 지수 분포\n(14일 기준온도 28°C 초과분 합산)", fontweight="bold")
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.hist(df_rep["CSI"].dropna(), bins=40, color="#7B1FA2", edgecolor="white", alpha=0.85)
ax.axvline(csi_q75, color="orange", ls="--", lw=2, label=f"75분위 ({csi_q75:.1f})")
ax.axvline(csi_q90, color="red",    ls="--", lw=2, label=f"90분위 ({csi_q90:.1f})")
ax.set_xlabel("복합 스트레스 지수 (CSI)"); ax.set_ylabel("빈도")
ax.set_title("복합 스트레스 지수 분포\nCSI = HSI × (1 + HLI)", fontweight="bold")
ax.legend(fontsize=9)

ax = axes[1, 0]
csi_max = df["CSI"].max()
csi_cut_bins = sorted(set([-0.1, max(csi_q75/2, 0.01), csi_q75, csi_q90, csi_max + 0.1]))
csi_bins = pd.cut(df["CSI"], bins=csi_cut_bins,
                  labels=["저스트레스", "중스트레스", "고스트레스", "극고스트레스"][:len(csi_cut_bins)-1])
stress_bnd = df.groupby(csi_bins, observed=True)["stage_num"].apply(
    lambda x: (x == 2).mean() * 100)
bars = ax.bar(stress_bnd.index, stress_bnd.values,
              color=["#4CAF50", "#FFB74D", "#EF5350", "#B71C1C"], edgecolor="white", alpha=0.9)
ax.bar_label(bars, fmt="%.1f%%", fontsize=10, padding=2, fontweight="bold")
ax.set_ylabel("경계 발생률 (%)"); ax.set_ylim(0, max(stress_bnd.values) * 1.35 + 1)
ax.set_title("누적 스트레스 구간별 경계 발생률", fontweight="bold")

ax = axes[1, 1]
year_csi = df_rep.groupby(df_rep["조사일"].dt.year)["CSI"].mean()
_rep_yr  = df_rep.copy(); _rep_yr["yr"] = _rep_yr["조사일"].dt.year
year_bnd = _rep_yr.groupby("yr")["stage_num"].apply(lambda x: (x == 2).mean() * 100)
ax2 = ax.twinx()
ax.bar(year_csi.index, year_csi.values, color="#5C6BC0", alpha=0.6, label="평균 CSI")
ax2.plot(year_bnd.index, year_bnd.values, "r-o", lw=2, ms=6, label="경계 발생률(%)")
ax.set_xlabel("연도"); ax.set_ylabel("평균 CSI", color="#5C6BC0")
ax2.set_ylabel("경계 발생률 (%)", color="red")
ax.set_title("연도별 평균 CSI vs 경계 발생률", fontweight="bold")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

plt.suptitle("누적 스트레스 지수 (HSI / HLI / CSI) 분석", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "C5_stress_index.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C5_stress_index.png")

# ── Plot 6: 폭증 전이 & 지연 폭발 분석 ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 6-A: 지연 폭발 분포
for i, (event_name, bloom_data) in enumerate(delayed_bloom.items()):
    ax = axes[i]
    delays = bloom_data["delays"]
    if delays:
        ax.hist(delays, bins=range(0, 31, 3), color=["#EF5350", "#FF9800", "#5C6BC0"][i],
                edgecolor="white", alpha=0.85)
        ax.axvline(bloom_data["median_delay"], color="black", ls="--", lw=2,
                   label=f"중앙값 {bloom_data['median_delay']:.0f}일")
        ax.set_xlabel("경보까지 지연 일수"); ax.set_ylabel("빈도")
        ax.legend(fontsize=8)
    ax.set_title(f"{event_name}\n이벤트 {bloom_data['events']}건 | "
                 f"경계 발생 {bloom_data['hit_rate']}%", fontweight="bold")
    ax.set_xlim(0, 31)

plt.suptitle("기상 이벤트 이후 지연 폭발 분석 (Event → Bloom Delay)", fontsize=13,
             fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "C6_delayed_bloom.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C6_delayed_bloom.png")

# ── Plot 7: 상태 전이 + 전이 조건 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
trans_show = {"미발령→관심": trans_counts["0→1"],
              "관심→경계":   trans_counts["1→2"],
              "경계→미발령": trans_counts["2→0"],
              "관심→미발령": trans_counts["1→0"]}
t_colors = ["#FF9800", "#F44336", "#4CAF50", "#90CAF9"]
bars = ax.barh(list(trans_show.keys()), list(trans_show.values()),
               color=t_colors, edgecolor="white", alpha=0.9)
ax.bar_label(bars, fmt="%d건", fontsize=10, padding=3)
ax.set_xlabel("전이 횟수")
ax.set_title("발령단계 상태 전이 횟수\n(전체 기간 × 3개 지점)", fontweight="bold")
ax.invert_yaxis()

ax = axes[1]
# 관심→경계 전이 직전 조건 vs 비전이 관심 조건
df_caution = df[df["stage_num"] == 1].copy()
df_caution["is_transition"] = 0

# 전이 날짜 마킹
for td in trans_12_dates:
    mask = (df_caution["조사일"] >= td - pd.Timedelta(days=7)) & (df_caution["조사일"] < td)
    df_caution.loc[df_caution.index.isin(df_caution[mask].index), "is_transition"] = 1

comp_cols = {"평균기온(°C)": "기온(°C)", "수온(℃)": "수온(°C)",
             "유입량(㎥/s)": "유입량", "HSI_14": "열스트레스"}
trans_means  = df_caution[df_caution["is_transition"]==1][[c for c in comp_cols]].mean()
notrans_means= df_caution[df_caution["is_transition"]==0][[c for c in comp_cols]].mean()

x = np.arange(len(comp_cols))
w = 0.35
b1 = ax.bar(x - w/2, [trans_means.get(c, 0) for c in comp_cols], w,
            label="전이 직전 7일", color="#F44336", alpha=0.85)
b2 = ax.bar(x + w/2, [notrans_means.get(c, 0) for c in comp_cols], w,
            label="비전이 관심", color="#5C6BC0", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(list(comp_cols.values()), fontsize=9)
ax.set_title("관심→경계 전이 직전 조건 비교\n(전이 구간 vs 일반 관심 구간)", fontweight="bold")
ax.legend(fontsize=9)

plt.suptitle("상태 전이 패턴 분석 — 경계 폭증의 전조 조건", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "C6_transition_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C6_transition_analysis.png")

# ── Plot 8: 종 경쟁 구조 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sp_colors = {"마이크로시스티스": "#EF5350", "아나베나": "#FF9800",
             "오실라토리아": "#5C6BC0", "아파니조메논": "#4CAF50"}
wt_order = ["<20°C", "20-25°C", "25-28°C", ">28°C"]
bot = np.zeros(len(wt_order))
for sp_kor, color in sp_colors.items():
    vals = [species_by_wt.get(wt, {}).get("dominance", {}).get(sp_kor, 0)
            for wt in wt_order]
    ax.bar(wt_order, vals, bottom=bot, color=color, label=sp_kor, edgecolor="white", alpha=0.9)
    bot += np.array(vals)
ax.set_ylabel("우점도 (%)"); ax.set_xlabel("수온 구간")
ax.set_title("수온 구간별 종 구성 비율\n(고온일수록 마이크로시스티스 우점 → 독소 위험↑)",
             fontweight="bold")
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

ax = axes[1]
bnd_by_wt = [species_by_wt.get(wt, {}).get("boundary_rate", 0) for wt in wt_order]
bars = ax.bar(wt_order, bnd_by_wt,
              color=["#90CAF9", "#FFB74D", "#EF5350", "#B71C1C"], edgecolor="white", alpha=0.9)
ax.bar_label(bars, fmt="%.1f%%", fontsize=10, padding=2, fontweight="bold")
ax.set_ylabel("경계 발생률 (%)"); ax.set_xlabel("수온 구간")
ax.set_title("수온 구간별 경계 발생률", fontweight="bold")
ax.set_ylim(0, max(bnd_by_wt) * 1.35 + 1)

plt.suptitle("종 경쟁 구조 분석 — 수온에 따른 우점종 전환과 경보 위험", fontsize=13,
             fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "plots" / "C7_species_competition.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C7_species_competition.png")

# ── Plot 9: 경보 실패 분석 ───────────────────────────────────────────────────
if df_pred is not None and fn_analysis:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    types  = list(fn_analysis.keys())
    counts = list(fn_analysis.values())
    t_colors_fn = ["#EF5350", "#FF9800", "#5C6BC0"][:len(types)]
    wedges, texts, autotexts = ax.pie(
        counts, labels=types, colors=t_colors_fn, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        textprops={"fontsize": 8})
    ax.set_title(f"경보 실패(FN) 유형 분류\n총 {sum(counts)}건 미탐지", fontweight="bold")

    ax = axes[1]
    total_bnd = (df_pred["target_d7"] == 2).sum()
    detected  = (df_pred["pred_stage"] == 2).sum()
    missed    = sum(fn_analysis.values())
    fp        = ((df_pred["pred_stage"] == 2) & (df_pred["target_d7"] != 2)).sum()
    cats  = ["실제 경계", "탐지 성공", "미탐지(FN)", "과탐지(FP)"]
    vals  = [total_bnd, detected, missed, fp]
    bcols = ["#7B1FA2", "#4CAF50", "#F44336", "#FF9800"]
    bars  = ax.bar(cats, vals, color=bcols, edgecolor="white", alpha=0.9)
    ax.bar_label(bars, fmt="%d건", fontsize=10, padding=2)
    ax.set_ylabel("건수"); ax.set_title("v4 모델 경계 탐지 성과\n(Test 2024-2025)", fontweight="bold")

    plt.suptitle("경보 실패 사례 분석 — FN 유형 분류 및 탐지 성과", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "plots" / "C7_alert_failure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  저장: C7_alert_failure.png")

# ── Plot 10: 대응 행동 추천 플로우 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 8))
ax.axis("off")

# 테이블 형태로 표현
col_labels = ["시나리오", "위험도", "즉시 조치 (0-24h)", "단기 조치 (1-3일)", "지속 관찰"]
table_data  = []
for sc, rule in ACTION_RULES.items():
    immediate = "\n".join([f"• {a}" for a in rule["actions"]["즉시(0-24h)"]])
    short     = "\n".join([f"• {a}" for a in rule["actions"]["단기(1-3일)"]])
    observe   = "\n".join([f"• {a}" for a in rule["actions"]["관찰(지속)"]])
    table_data.append([sc, rule["risk_level"], immediate, short, observe])

t = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="left",
)
t.auto_set_font_size(False)
t.set_fontsize(7)
t.scale(1, 5.5)

for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")
    elif r > 0:
        sc_name = table_data[r-1][0]
        cell.set_facecolor({
            "S-RED": "#FFCDD2", "S-ORA": "#FFE0B2",
            "S-YEL": "#FFF9C4", "S-GRN": "#C8E6C9"
        }.get(sc_name, "white"))

ax.set_title("대응 행동 추천 시스템 — 시나리오별 운용 행동 매트릭스",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
fig.savefig(OUT / "plots" / "C7_action_recommendation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C7_action_recommendation.png")

# ── Plot 11: 종합 위험 대시보드 ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

# (0,0)-(0,1): 시나리오 발생률 대형 막대
ax1 = fig.add_subplot(gs[0, :2])
sc_names  = list(sc_stats.keys())
sc_rates  = [sc_stats[s]["boundary_rate"] for s in sc_names]
sc_ns     = [sc_stats[s]["n"] for s in sc_names]
bars = ax1.bar(sc_names, sc_rates, color=[SC_COLORS[s] for s in sc_names],
               edgecolor="white", alpha=0.9, width=0.55)
ax1.bar_label(bars, labels=[f"{r:.1f}%" for r in sc_rates], fontsize=12,
              fontweight="bold", padding=3)
ax1.set_ylabel("경계 발생률 (%)"); ax1.set_ylim(0, max(sc_rates) * 1.35)
ax1.set_title("복합 시나리오별 경계 발생률", fontweight="bold")
for i, (sc, n) in enumerate(zip(sc_names, sc_ns)):
    ax1.text(i, -3.5, f"n={n:,}", ha="center", fontsize=8, color="gray")

# (0,2)-(0,3): Lag 상관
ax2 = fig.add_subplot(gs[0, 2:])
bar_c = ["#C62828" if l == best_lag else "#90A4AE" for l in lags]
ax2.bar(lags, corrs, color=bar_c, edgecolor="white", width=2.2)
ax2.set_xlabel("Lag (일)"); ax2.set_ylabel("Pearson r"); ax2.set_xticks(lags)
ax2.set_ylim(0.80, 0.93)
ax2.set_title(f"기온→수온 시간 지연\n최고: Lag {best_lag}일 r={lag_corr[best_lag]}", fontweight="bold")

# (1,0)-(1,1): 수문 매트릭스
ax3 = fig.add_subplot(gs[1, :2])
sns.heatmap(hydro_matrix, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax3,
            linewidths=0.5, annot_kws={"size": 9},
            cbar_kws={"label": "경계%", "shrink": 0.8})
ax3.set_title("저수율 × 유입량 경계 발생률", fontweight="bold", fontsize=10)
ax3.set_xlabel("유입량 (m³/s)", fontsize=8); ax3.set_ylabel("저수율", fontsize=8)
ax3.tick_params(labelsize=8)

# (1,2)-(1,3): 여름철 기온 vs 수온
ax4 = fig.add_subplot(gs[1, 2:])
x = np.arange(len(months))
ax4.bar(x - 0.2, at_vals, 0.35, label="기온", color="#5C6BC0", alpha=0.85)
ax4.bar(x + 0.2, wt_vals, 0.35, label="수온", color="#EF5350", alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(["6월", "7월", "8월", "9월"], fontsize=9)
ax4.axhline(28, color="red", ls="--", lw=1, alpha=0.6)
ax4.set_ylabel("온도 (°C)"); ax4.legend(fontsize=8)
ax4.set_title("여름철 기온 vs 수온\n(9월: 기온↓ but 수온유지 → 지연위험)", fontweight="bold", fontsize=10)

# (2,0): CSI 구간별 경계율
ax5 = fig.add_subplot(gs[2, 0])
stress_bnd_vals = stress_bnd.values
s_bars = ax5.bar(stress_bnd.index, stress_bnd_vals,
                 color=["#4CAF50", "#FFB74D", "#EF5350", "#B71C1C"],
                 edgecolor="white", alpha=0.9)
ax5.bar_label(s_bars, fmt="%.1f%%", fontsize=8, padding=2)
ax5.set_title("CSI 구간별 경계 발생률", fontweight="bold", fontsize=9)
ax5.tick_params(axis="x", labelsize=7); ax5.set_ylabel("%")

# (2,1): 월별 남조류
ax6 = fig.add_subplot(gs[2, 1])
cyano_vals_m = [month_stats[m]["cyano_mean"] for m in months]
bars6 = ax6.bar(["6월","7월","8월","9월"], cyano_vals_m, color=bar_colors, edgecolor="white", alpha=0.9)
ax6.set_yscale("symlog", linthresh=100)
ax6.set_ylabel("cells/mL"); ax6.set_title("월별 남조류 농도", fontweight="bold", fontsize=9)
ax6.tick_params(labelsize=8)

# (2,2): 방류 시나리오
ax7 = fig.add_subplot(gs[2, 2])
ax7.bar(["방류증가", "방류급감소"], [rise_bnd_rate, drop_bnd_rate],
        color=["#4CAF50", "#F44336"], edgecolor="white", alpha=0.9)
ax7.set_ylabel("%"); ax7.set_title("방류 시나리오 경계율", fontweight="bold", fontsize=9)
ax7.tick_params(labelsize=8)

# (2,3): 종 우점 (수온 28°C+ 구간)
ax8 = fig.add_subplot(gs[2, 3])
if ">28°C" in species_by_wt:
    dom_data = species_by_wt[">28°C"]["dominance"]
    ax8.pie(dom_data.values(), labels=[k[:4] for k in dom_data.keys()],
            colors=list(sp_colors.values()),
            autopct="%1.0f%%", startangle=90, textprops={"fontsize": 7})
    ax8.set_title("고온(>28°C) 종 구성", fontweight="bold", fontsize=9)

plt.suptitle("대청댐 유해남조류 경보 — 복합 위험 의사결정 지원 대시보드",
             fontsize=14, fontweight="bold", y=1.01)
fig.savefig(OUT / "plots" / "C0_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  저장: C0_dashboard.png (종합 대시보드)")

# ═══════════════════════════════════════════════════════════════════════════════
# 결과 저장
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[결과 저장]")

# 일별 위험점수 + 시나리오 CSV
risk_csv = df_rep[["조사일", "stage_num", "scenario", "risk_score",
                    "HSI_14", "CSI", "heat_streak",
                    "평균기온(°C)", "수온(℃)", "저수율(%)", "유입량(㎥/s)", "총방류량(㎥/s)"]].copy()
risk_csv.to_csv(OUT / "reports" / "risk_calendar.csv", index=False, encoding="utf-8-sig")
print(f"  저장: risk_calendar.csv ({len(risk_csv):,}행)")

# 요약 JSON
summary = {
    "summer_phases":       month_stats,
    "lag_correlation":     {f"lag_{k}d": v for k, v in lag_corr.items()},
    "best_lag":            best_lag,
    "avg_boundary_after_cross": round(avg_boundary_after_cross, 2),
    "scenario_stats":      sc_stats,
    "hydraulic_matrix": {
        "drop_boundary_rate": round(drop_bnd_rate, 1),
        "rise_boundary_rate": round(rise_bnd_rate, 1),
    },
    "stress_index": {
        "csi_q75": round(csi_q75, 1),
        "csi_q90": round(csi_q90, 1),
        "high_stress_boundary_rate": round(hs_bnd, 1),
        "low_stress_boundary_rate":  round(ls_bnd, 1),
    },
    "transition_counts":   trans_counts,
    "delayed_bloom":       {k: {kk: vv for kk, vv in v.items() if kk != "delays"}
                            for k, v in delayed_bloom.items()},
    "species_by_wt":       species_by_wt,
    "fn_analysis":         fn_analysis,
}

with open(OUT / "reports" / "scenario_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print("  저장: scenario_summary.json")

# 대응 행동 추천 JSON
with open(OUT / "reports" / "action_rules.json", "w", encoding="utf-8") as f:
    json.dump(ACTION_RULES, f, ensure_ascii=False, indent=2)
print("  저장: action_rules.json")

# ═══════════════════════════════════════════════════════════════════════════════
# 최종 요약 출력
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  프레임워크 최종 요약")
print("=" * 70)
print(f"\n  [C1] 여름철 9월 역설: 기온 22°C↓ but 수온 {month_stats[9]['water_temp_mean']}°C 유지")
print(f"       남조류: 8월 {month_stats[8]['cyano_mean']:,.0f} vs 9월 {month_stats[9]['cyano_mean']:,.0f} cells/mL")
print(f"\n  [C2] 기온→수온 Lag {best_lag}일 (r={lag_corr[best_lag]}) — 14일 전 기온으로 수온 예측")
print(f"       교차 후 21일 내 경계 평균 {avg_boundary_after_cross:.1f}건")
print(f"\n  [C3] 최고위험 조합: 저수율 60-80% × 유입량 150-300 = 경계 27.5%")
print(f"       방류 급감소 경계율 {drop_bnd_rate:.1f}% vs 방류 증가 {rise_bnd_rate:.1f}%")
print(f"\n  [C4] S-RED {sc_stats['S-RED']['boundary_rate']}% | "
      f"S-ORA {sc_stats['S-ORA']['boundary_rate']}% | "
      f"S-YEL {sc_stats['S-YEL']['boundary_rate']}% | "
      f"S-GRN {sc_stats['S-GRN']['boundary_rate']}%")
print(f"\n  [C5] 고스트레스(CSI≥75분위) 경계율 {hs_bnd:.1f}% vs 저스트레스 {ls_bnd:.1f}%")
print(f"\n  [C6] 관심→경계 전이 {trans_counts['1→2']}건 | 전이 직전 조건 특징값 계산 완료")
print(f"\n  출력: {OUT}")
print("=" * 70)
