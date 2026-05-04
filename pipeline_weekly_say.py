#!/usr/bin/env python3
"""
finaldata_say.csv 전용 — 과제 학습 구조에 맞춘 주 단위 파이프라인

- 입력은 **항상 `finaldata_say.csv`만** (일 D+7은 `pipeline_daily_d7.py` 참고).
- 최근 4주(동일 지점) 요약 → **다음 주** total_cyano(max/mean), 발령단계 최고치, 관심 이상 여부
- 회귀(log1p) + 단계 분류 + 이진(관심 이상): **LightGBM** 주력, **XGBoost** 이진 앙상블·단계 앙상블(선택)
- **확률 보정**(isotonic, 학습 구간 CV) 이진 관심 이상
- **순 외생 특성** 실험(세포수·종·발령 관련 lw* 열 제외) 지표 비교
- 운영 지표: 관심 이상 **재현율·정밀도**, 경계 단계 **F1**
- SHAP: 전체 + (검증 중) 관심 이상/미만 **부분집합** 각 1장
- 시차 힌트: `feature_importances_` 상위 태그(lw1 vs lw4_mean vs lw4_slope)
- 규칙 기반 시나리오 권고(JSON)
- LSTM은 포함하지 않음: 다음 주 log1p(주간 max 세포수) 시퀀스 회귀는
  `pipeline_weekly_extras.py` → `outputs/weekly_extras/extras_metrics.json` 의 lstm_* 지표만 참고.

**Ollama와 예측 모델의 관계**
- 수치 예측은 **LightGBM / XGBoost** 트리 모델. Ollama는 자연어 다듬기만.
- Ollama(LLM)는 **선택 사항**으로, 이미 나온 규칙 JSON·확률을 **보고서/대시용 자연어 문단**으로만 다듬을 때 쓴다.
  (의사결정 지원 “설명 문구” 생성 — 정확도·적중률 평가 대상이 아님. 제출물에 필수 아님.)

실행: python3 pipeline_weekly_say.py  (권장: /usr/bin/python3.10)
옵션: OLLAMA_NARRATIVE=1 OLLAMA_MODEL=llama3.2 OLLAMA_HOST=http://127.0.0.1:11434
"""
from __future__ import annotations

import json
import os
import urllib.request
import warnings
from pathlib import Path

from paths import OUT_WEEKLY_SAY as OUT_DIR, SAY_CSV

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_config import setup_korean_matplotlib
import pandas as pd
from collections import Counter

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

STAGE_NAMES = ["미발령", "관심", "경계", "조류대발생"]

def _read_say() -> pd.DataFrame:
    if not SAY_CSV.exists():
        raise FileNotFoundError(f"먼저 finaldata_say.csv 가 필요합니다: {SAY_CSV}")
    df = pd.read_csv(SAY_CSV, encoding="utf-8-sig")
    df["조사일"] = pd.to_datetime(df["조사일"], errors="coerce")
    df = df.dropna(subset=["조사일", "채수위치"])
    num_cols = [
        "total_cyano",
        "microcystis",
        "anabaena",
        "oscillatoria",
        "aphanizomenon",
        "발령단계_코드",
        "수온(℃)",
        "pH",
        "DO(㎎/L)",
        "탁도",
        "Chl-a (㎎/㎥)",
        "투명도",
        "합계 일조시간(hr)",
        "유입량(㎥/s)",
        "총방류량(㎥/s)",
        "수위(EL.m)",
        "저수율(%)",
        "일강수량(mm)",
        "강우량(mm)",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "일일강수합_mm" not in df.columns:
        r = df["일강수량(mm)"].fillna(0) if "일강수량(mm)" in df.columns else 0
        if "강우량(mm)" in df.columns:
            r = df["일강수량(mm)"].fillna(0) + df["강우량(mm)"].fillna(0)
        df["일일강수합_mm"] = r
    return df


def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = df["조사일"].dt.to_period("W-SUN")
    df["week_start"] = df["week"].apply(lambda p: p.start_time)
    gcols = ["채수위치", "week", "week_start"]

    agg_kw: dict = {}
    if "total_cyano" in df.columns:
        agg_kw["total_cyano_mean"] = ("total_cyano", "mean")
        agg_kw["total_cyano_max"] = ("total_cyano", "max")
    if "발령단계_코드" in df.columns:
        agg_kw["발령단계_코드_max"] = ("발령단계_코드", "max")
    for c in [
        "microcystis",
        "anabaena",
        "oscillatoria",
        "aphanizomenon",
        "수온(℃)",
        "pH",
        "DO(㎎/L)",
        "탁도",
        "Chl-a (㎎/㎥)",
        "투명도",
        "합계 일조시간(hr)",
        "유입량(㎥/s)",
        "총방류량(㎥/s)",
        "수위(EL.m)",
        "저수율(%)",
    ]:
        if c in df.columns:
            agg_kw[f"{c}_mean"] = (c, "mean")
    if "일일강수합_mm" in df.columns:
        agg_kw["일강수_주합_mm"] = ("일일강수합_mm", "sum")

    w = df.groupby(gcols, as_index=False).agg(**agg_kw)
    return w.sort_values(["채수위치", "week_start"]).reset_index(drop=True)


def build_weekly_supervised(df: pd.DataFrame) -> pd.DataFrame:
    weekly = daily_to_weekly(df)
    all_rows: list[dict] = []
    for site, sub in weekly.groupby("채수위치"):
        sub = sub.sort_values("week_start").reset_index(drop=True)
        cols_track = [c for c in sub.columns if c not in ("채수위치", "week", "week_start")]

        for k in range(4, len(sub)):
            past = sub.iloc[k - 4 : k]
            nxt = sub.iloc[k]
            row: dict = {
                "채수위치": str(site),
                "week_start_target": nxt["week_start"],
            }
            for c in cols_track:
                s = pd.to_numeric(past[c], errors="coerce")
                row[f"lw4_mean_{c}"] = float(s.mean())
                row[f"lw4_std_{c}"] = float(s.std(ddof=0)) if len(s) > 1 else 0.0
                row[f"lw4_slope_{c}"] = float(s.iloc[-1] - s.iloc[0])

            # 최근 2주·1주 핵심 요약 (목차: 1·2·4주 영향)
            keyw = [
                "total_cyano_max",
                "total_cyano_mean",
                "발령단계_코드_max",
                "수온(℃)_mean",
                "Chl-a (㎎/㎥)_mean",
                "유입량(㎥/s)_mean",
                "총방류량(㎥/s)_mean",
                "일강수_주합_mm",
                "합계 일조시간(hr)_mean",
            ]
            past2 = sub.iloc[k - 2 : k]
            past1 = sub.iloc[k - 1 : k]
            for c in keyw:
                if c not in past2.columns:
                    continue
                s2 = pd.to_numeric(past2[c], errors="coerce")
                s1 = pd.to_numeric(past1[c], errors="coerce")
                row[f"lw2_mean_{c}"] = float(s2.mean())
                row[f"lw1_{c}"] = float(s1.iloc[-1]) if len(s1) else np.nan

            row["y_cyano_max_next"] = float(nxt["total_cyano_max"]) if "total_cyano_max" in nxt else np.nan
            row["y_cyano_mean_next"] = float(nxt["total_cyano_mean"]) if "total_cyano_mean" in nxt else np.nan
            row["y_stage_max_next"] = int(float(nxt["발령단계_코드_max"])) if "발령단계_코드_max" in nxt else 0
            row["y_alert_ge1_next"] = int(row["y_stage_max_next"] >= 1)
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    out["week_start_target"] = pd.to_datetime(out["week_start_target"])
    return out


def feature_cols(tbl: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in tbl.columns if c.startswith("lw4_") or c.startswith("lw2_") or c.startswith("lw1_")]
    )


def exogenous_feature_indices(feat_all: list[str]) -> list[int]:
    """세포수·4종·발령단계 관련 lag 특성 제외 → 기상·수문·수집(Chl-a 등) 위주."""
    banned = (
        "total_cyano",
        "microcystis",
        "anabaena",
        "oscillatoria",
        "aphanizomenon",
        "발령단계",
    )
    return [i for i, f in enumerate(feat_all) if not any(b in f for b in banned)]


def lag_importance_hint(feat_all: list[str], importances: np.ndarray, top_k: int = 10) -> dict[str, str]:
    """상위 특성을 lw1(직전주)·lw2·lw4_mean·lw4_slope 로 묶어 보고서용 한 줄 요약."""
    order = np.argsort(importances)[::-1][:top_k]
    bucket: list[str] = []
    for i in order:
        f = feat_all[i]
        if f.startswith("lw1_"):
            bucket.append("직전1주(lw1)")
        elif f.startswith("lw2_"):
            bucket.append("최근2주평균(lw2)")
        elif "lw4_slope" in f or ("lw4_" in f and "slope" in f):
            bucket.append("4주추세(slope)")
        elif f.startswith("lw4_mean"):
            bucket.append("4주평균")
        else:
            bucket.append("기타")
    cnt = Counter(bucket)
    summary = ", ".join(f"{k} {v}회" for k, v in cnt.most_common())
    top_names = [feat_all[i] for i in order[:5]]
    return {"요약": summary, "상위5특성": ", ".join(top_names)}


def time_mask(dates: pd.Series, holdout: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    u = np.sort(dates.dropna().unique())
    cut = u[max(int(len(u) * (1.0 - holdout)), 1) - 1]
    return (dates <= cut).values, (dates > cut).values


def shap_bar(model, X: np.ndarray, feat_names: list[str], path: Path, title: str) -> None:
    try:
        import shap

        setup_korean_matplotlib()
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            stacked = np.mean(np.stack([np.abs(s) for s in sv], axis=0), axis=0)
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            stacked = np.abs(sv).mean(axis=2)
        else:
            stacked = np.abs(np.asarray(sv))
        arr = stacked.mean(axis=0)
        order = np.argsort(arr)[::-1][:28]
        plt.figure(figsize=(10, 8))
        plt.barh(np.array(feat_names)[order][::-1], arr[order][::-1])
        plt.xlabel("mean |SHAP|")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception as e:
        (path.parent / (path.stem + "_error.txt")).write_text(str(e), encoding="utf-8")


def scenario_rules(
    row: pd.Series,
    p_alert: float,
    pred_cyano: float,
    ref_inflow_median: float = np.nan,
) -> list[dict]:
    """규칙 기반 권고(시나리오 1~4 + 일반)."""
    scenarios: list[dict] = []

    def add(sid: str, title: str, cond: str, expect: str, actions: list[str]) -> None:
        scenarios.append(
            {"id": sid, "title": title, "조건": cond, "예상": expect, "권고": actions}
        )

    tmean = row.get("lw4_mean_수온(℃)_mean", np.nan)
    sslope = row.get("lw4_slope_합계 일조시간(hr)_mean", np.nan)
    rslope = row.get("lw4_slope_일강수_주합_mm", np.nan)
    bslope = row.get("lw4_slope_총방류량(㎥/s)_mean", np.nan)
    chla_slope = row.get("lw4_slope_Chl-a (㎎/㎥)_mean", np.nan)
    ph_slope = row.get("lw4_slope_pH_mean", np.nan)
    turb_slope = row.get("lw4_slope_탁도_mean", np.nan)
    inflow_mean = row.get("lw4_mean_유입량(㎥/s)_mean", np.nan)

    if pd.notna(tmean) and tmean > 22 and pd.notna(sslope) and sslope > 0 and pd.notna(bslope) and bslope < 0:
        add(
            "S1",
            "폭염·고일조·방류 감소",
            "최근 4주 평균 수온 높음, 일조 증가 추세, 방류 감소 추세",
            "정체·광합성 유리 → 남조류 증가 가능성",
            [
                "관심 단계 사전 대응 검토",
                "현장 채수·종조성 모니터링 강화",
                "취수장·정수 약품 투입 계획 점검",
            ],
        )

    hi_inflow = (
        pd.notna(inflow_mean)
        and pd.notna(ref_inflow_median)
        and inflow_mean > ref_inflow_median * 1.1
    )
    if pd.notna(rslope) and rslope > 20 and hi_inflow:
        add(
            "S2",
            "집중호우·유입 증가",
            "최근 4주 강수·유입 증가 추세",
            "단기 혼합으로 세포 수 변동 가능, 영양염 유입에 따른 지연 증가 위험",
            [
                "1~2주 후 재예측·채수 주기 단축",
                "탁도·유해남조류 동시 추적",
            ],
        )

    if pd.notna(bslope) and bslope < 0 and pd.notna(tmean) and tmean > 20:
        add(
            "S3",
            "방류 감소·수온 높음",
            "방류 감소 추세와 고수온",
            "체류시간 증가로 조류 증식 가능성",
            ["수문·방류 운영 검토", "취수 수심·위치 조정 가능성 검토"],
        )

    if pd.notna(chla_slope) and chla_slope > 0 and pd.notna(ph_slope) and ph_slope > 0:
        add(
            "S4",
            "Chl-a·pH 상승",
            "클로로필-a 및 pH 상승 추세",
            "조류 생체량 증가 신호, 유해남조류 증가 가능성",
            ["조류 종 분석 강화", "발령단계 전환 모니터링"],
        )

    if p_alert >= 0.45:
        add(
            "SX",
            "모델 기반 위험",
            f"관심 이상 확률 약 {p_alert:.0%}, 예측 세포수(대략) {pred_cyano:.0f} cells/mL",
            "다음 주 관심 이상 가능성 상대적으로 큼",
            [
                "해당 지점 수질 모니터링 강화",
                "취수장 사전 대응·활성탄 등 준비",
                "방류량 조정 가능성 검토",
            ],
        )

    if not scenarios:
        add("S0", "특이 패턴 없음", "주요 시나리오 조건 미충족", "상시 모니터링 유지", ["정기 관측 유지"])

    return scenarios


def ollama_narrative(scenarios: list[dict], site: str, week: str) -> str | None:
    if os.environ.get("OLLAMA_NARRATIVE", "").lower() not in ("1", "true", "yes"):
        return None
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    prompt = (
        "당신은 대청댐 수질·조류 담당 전문가입니다. 아래 JSON 시나리오를 바탕으로 "
        f"지점={site}, 대상주={week} 에 대한 5문장 이내 한국어 운영 권고문을 작성하세요. "
        "실제 조치는 현장 규정을 따릅니다.\n\n"
        + json.dumps(scenarios, ensure_ascii=False)
    )
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data.get("response", "").strip()
    except Exception as e:
        return f"[Ollama 호출 실패: {e}]"


def main() -> None:
    setup_korean_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _read_say()
    tbl = build_weekly_supervised(df)
    tbl = tbl.dropna(subset=["y_cyano_max_next", "y_stage_max_next"]).reset_index(drop=True)
    tbl.to_csv(OUT_DIR / "weekly_supervised_rows.csv", index=False, encoding="utf-8-sig")

    feats = feature_cols(tbl)
    tbl[feats] = tbl[feats].replace([np.inf, -np.inf], np.nan).fillna(tbl[feats].median(numeric_only=True))

    ref_inflow_m = float(np.nanmedian(tbl["lw4_mean_유입량(㎥/s)_mean"].values)) if "lw4_mean_유입량(㎥/s)_mean" in tbl.columns else np.nan

    le = pd.factorize(tbl["채수위치"])[0]
    tbl["site_code"] = le
    feat_all = feats + ["site_code"]

    X = tbl[feat_all].values
    y_stage = tbl["y_stage_max_next"].clip(0, 3).astype(int).values
    y_bin = tbl["y_alert_ge1_next"].values
    y_reg = np.log1p(np.maximum(tbl["y_cyano_max_next"].values, 0.0))

    dates = tbl["week_start_target"]
    tr, te = time_mask(dates, 0.15)

    X_tr, X_te = X[tr], X[te]
    meta_te = tbl.loc[te].reset_index(drop=True)

    metrics: dict = {}

    # --- 회귀: 다음 주 log1p(주간 max 세포 수)
    try:
        from lightgbm import LGBMRegressor

        reg = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=48,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
        reg.fit(X_tr, y_reg[tr])
        pred_log_te = reg.predict(X_te)
        y_te_reg = y_reg[te]
        metrics["regression_log1p_cyano"] = {
            "r2": float(r2_score(y_te_reg, pred_log_te)),
            "mae_log": float(mean_absolute_error(y_te_reg, pred_log_te)),
            "mae_cells_approx": float(
                mean_absolute_error(np.expm1(y_te_reg), np.maximum(np.expm1(pred_log_te), 0.0))
            ),
        }
        shap_bar(reg, X_te[: min(400, len(X_te))], feat_all, OUT_DIR / "shap_weekly_regression.png", "주간 모델 SHAP — log1p(세포수)")
    except Exception as e:
        metrics["regression_error"] = str(e)
        reg = None
        pred_log_te = np.zeros(X_te.shape[0])

    pred_st_report = np.zeros(X_te.shape[0], dtype=int)

    # --- 분류: 다음 주 최고 발령단계
    try:
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=48,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        clf.fit(X_tr, y_stage[tr])
        pred_st = clf.predict(X_te)
        y_te_st = y_stage[te]
        present = sorted(set(y_te_st.tolist()) | set(pred_st.tolist()))
        names = [STAGE_NAMES[i] for i in range(max(present) + 1)]
        metrics["classification_stage"] = {
            "accuracy": float(accuracy_score(y_te_st, pred_st)),
            "f1_macro": float(f1_score(y_te_st, pred_st, average="macro", zero_division=0)),
            "report": classification_report(y_te_st, pred_st, labels=list(range(len(names))), target_names=names, zero_division=0),
        }
        shap_bar(clf, X_te[: min(400, len(X_te))], feat_all, OUT_DIR / "shap_weekly_stage.png", "주간 모델 SHAP — 경보 단계")

        try:
            from xgboost import XGBClassifier

            nc = int(np.max(y_stage[tr])) + 1
            nc = max(nc, 3)
            clf_x = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=43,
                verbosity=0,
                objective="multi:softprob",
                num_class=nc,
            )
            clf_x.fit(X_tr, y_stage[tr])
            p_lgb_s = clf.predict_proba(X_te)
            p_xgb_s = clf_x.predict_proba(X_te)
            p_ens_s = (p_lgb_s + p_xgb_s) / 2.0
            pred_st_ens = np.argmax(p_ens_s, axis=1)
            metrics["classification_stage_ensemble_lgb_xgb"] = {
                "accuracy": float(accuracy_score(y_te_st, pred_st_ens)),
                "f1_macro": float(f1_score(y_te_st, pred_st_ens, average="macro", zero_division=0)),
            }
            pred_st_report = pred_st_ens
        except Exception as e2:
            metrics["classification_stage_ensemble_error"] = str(e2)
            pred_st_report = pred_st
    except Exception as e:
        metrics["classification_stage_error"] = str(e)
        clf = None
        pred_st = np.zeros(X_te.shape[0], dtype=int)
        pred_st_report = pred_st

    y_te_b = y_bin[te]

    # --- 이진: 관심 이상 (LGB + XGB 평균 확률, isotonic 보정, 순 외생 별도)
    p_bin = np.full(X_te.shape[0], 0.5)
    p_xgb_b = np.full(X_te.shape[0], 0.5)
    p_ens_b = np.full(X_te.shape[0], 0.5)
    p_cal = np.full(X_te.shape[0], 0.5)
    p_exo = np.full(X_te.shape[0], 0.5)

    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        binm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        binm.fit(X_tr, y_bin[tr])
        p_bin = binm.predict_proba(X_te)[:, 1]

        pos = max(1, int(y_bin[tr].sum()))
        neg = max(1, len(y_bin[tr]) - int(y_bin[tr].sum()))
        spw = neg / pos
        xgb_b = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=float(spw),
            random_state=43,
            verbosity=0,
        )
        xgb_b.fit(X_tr, y_bin[tr])
        p_xgb_b = xgb_b.predict_proba(X_te)[:, 1]
        p_ens_b = (p_bin + p_xgb_b) / 2.0

        base_cal = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=44,
            verbose=-1,
        )
        cal = CalibratedClassifierCV(base_cal, method="isotonic", cv=3)
        cal.fit(X_tr, y_bin[tr])
        p_cal = cal.predict_proba(X_te)[:, 1]

        exo_ix = exogenous_feature_indices(feat_all)
        if len(exo_ix) < 4:
            metrics["binary_exogenous_only_lgb"] = {"skipped": True, "n_features": len(exo_ix)}
        else:
            X_tr_e, X_te_e = X_tr[:, exo_ix], X_te[:, exo_ix]
            bin_e = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=45,
                verbose=-1,
            )
            bin_e.fit(X_tr_e, y_bin[tr])
            p_exo = bin_e.predict_proba(X_te_e)[:, 1]
            metrics["binary_exogenous_only_lgb"] = {
                "n_features": len(exo_ix),
                "roc_auc": float(roc_auc_score(y_te_b, p_exo)) if len(np.unique(y_te_b)) > 1 else None,
                "recall_alert": float(recall_score(y_te_b, (p_exo >= 0.5).astype(int), pos_label=1, zero_division=0)),
            }

        pred_b = (p_ens_b >= 0.5).astype(int)
        pred_b_cal = (p_cal >= 0.5).astype(int)

        metrics["binary_lgb"] = {
            "roc_auc": float(roc_auc_score(y_te_b, p_bin)) if len(np.unique(y_te_b)) > 1 else None,
            "accuracy": float(accuracy_score(y_te_b, (p_bin >= 0.5).astype(int))),
            "recall_alert": float(recall_score(y_te_b, (p_bin >= 0.5).astype(int), pos_label=1, zero_division=0)),
            "precision_alert": float(precision_score(y_te_b, (p_bin >= 0.5).astype(int), pos_label=1, zero_division=0)),
        }
        metrics["binary_xgb"] = {
            "roc_auc": float(roc_auc_score(y_te_b, p_xgb_b)) if len(np.unique(y_te_b)) > 1 else None,
        }
        metrics["binary_ensemble_avg_proba_lgb_xgb"] = {
            "roc_auc": float(roc_auc_score(y_te_b, p_ens_b)) if len(np.unique(y_te_b)) > 1 else None,
            "accuracy": float(accuracy_score(y_te_b, pred_b)),
            "f1": float(f1_score(y_te_b, pred_b, zero_division=0)),
            "recall_alert": float(recall_score(y_te_b, pred_b, pos_label=1, zero_division=0)),
            "precision_alert": float(precision_score(y_te_b, pred_b, pos_label=1, zero_division=0)),
        }
        metrics["binary_calibrated_isotonic_lgb"] = {
            "roc_auc": float(roc_auc_score(y_te_b, p_cal)) if len(np.unique(y_te_b)) > 1 else None,
            "accuracy": float(accuracy_score(y_te_b, pred_b_cal)),
            "f1": float(f1_score(y_te_b, pred_b_cal, zero_division=0)),
            "recall_alert": float(recall_score(y_te_b, pred_b_cal, pos_label=1, zero_division=0)),
            "precision_alert": float(precision_score(y_te_b, pred_b_cal, pos_label=1, zero_division=0)),
        }

        y_te_st = y_stage[te]
        mask_boundary_true = y_te_st == 2
        mask_boundary_pred = pred_st_report == 2
        metrics["operational_boundary_f1"] = float(
            f1_score(mask_boundary_true.astype(int), mask_boundary_pred.astype(int), zero_division=0)
        )

        metrics["시차_특성_힌트_이진LGB"] = lag_importance_hint(feat_all, binm.feature_importances_)

        if np.sum(y_te_b == 1) >= 12:
            ix1 = np.where(y_te_b == 1)[0][: min(350, np.sum(y_te_b == 1))]
            shap_bar(
                binm,
                X_te[ix1],
                feat_all,
                OUT_DIR / "shap_binary_subset_val_y1.png",
                "SHAP 이진(검증: 다음주 관심이상=실제1)",
            )
        if np.sum(y_te_b == 0) >= 12:
            ix0 = np.where(y_te_b == 0)[0][: min(350, np.sum(y_te_b == 0))]
            shap_bar(
                binm,
                X_te[ix0],
                feat_all,
                OUT_DIR / "shap_binary_subset_val_y0.png",
                "SHAP 이진(검증: 다음주 관심이상=실제0)",
            )
    except Exception as e:
        metrics["binary_error"] = str(e)

    if "binary_error" in metrics:
        p_scen = p_bin
    else:
        p_scen = p_cal

    # --- 검증 구간 행 단위 예측 (제출·보고용)
    hold_w = pd.DataFrame(
        {
            "week_start_target": pd.to_datetime(meta_te["week_start_target"]),
            "채수위치": meta_te["채수위치"].astype(str),
            "y_stage_max_next": meta_te["y_stage_max_next"].astype(int),
            "y_stage_name": [STAGE_NAMES[int(min(max(c, 0), 3))] for c in meta_te["y_stage_max_next"]],
            "y_alert_ge1_next": meta_te["y_alert_ge1_next"].astype(int),
            "y_cyano_max_next": meta_te["y_cyano_max_next"].astype(float),
            "pred_stage_code": pred_st_report.astype(int),
            "pred_stage_name": [STAGE_NAMES[int(min(max(c, 0), 3))] for c in pred_st_report],
            "pred_cyano_max_approx": np.expm1(np.maximum(pred_log_te, 0.0)),
            "p_alert_ge1_lgb": p_bin.astype(float),
            "p_alert_ge1_ensemble": p_ens_b.astype(float),
            "p_alert_ge1_calibrated": p_cal.astype(float),
            "pred_binary_ensemble_ge0.5": (p_ens_b >= 0.5).astype(int),
            "pred_binary_calibrated_ge0.5": (p_cal >= 0.5).astype(int),
        }
    )
    hold_w["holdout_note"] = "시간 홀드아웃 후반 15% (week_start_target 고유값 기준)"
    hold_w.to_csv(OUT_DIR / "holdout_predictions_weekly.csv", index=False, encoding="utf-8-sig")

    # --- 시나리오 JSON (검증 구간 샘플)
    recs: list[dict] = []
    for i in range(len(meta_te)):
        r = meta_te.iloc[i]
        scen = scenario_rules(
            r,
            float(p_scen[i]),
            float(np.expm1(pred_log_te[i])) if len(pred_log_te) > i else 0.0,
            ref_inflow_median=ref_inflow_m,
        )
        item = {
            "채수위치": r["채수위치"],
            "week_start_target": str(r["week_start_target"].date()),
            "p_alert_ge1_lgb": float(p_bin[i]) if len(p_bin) > i else None,
            "p_alert_ge1_ensemble": float(p_ens_b[i]) if len(p_ens_b) > i else None,
            "p_alert_ge1_calibrated": float(p_cal[i]) if len(p_cal) > i else None,
            "p_alert_ge1_scenario": float(p_scen[i]) if len(p_scen) > i else None,
            "pred_stage": int(pred_st_report[i]) if len(pred_st_report) > i else None,
            "pred_cyano_max_approx": float(np.expm1(pred_log_te[i])) if len(pred_log_te) > i else None,
            "scenarios": scen,
        }
        narr = ollama_narrative(scen, str(r["채수위치"]), str(r["week_start_target"].date()))
        if narr:
            item["ollama_narrative"] = narr
        recs.append(item)

    (OUT_DIR / "scenario_recommendations_holdout.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 시계열: 관심 이상 확률
    setup_korean_matplotlib()
    plt.figure(figsize=(12, 5))
    plt.plot(meta_te["week_start_target"], p_bin, label="LGB P(관심 이상)", alpha=0.85)
    plt.plot(meta_te["week_start_target"], p_ens_b, label="LGB+XGB 평균 확률", alpha=0.85)
    plt.plot(meta_te["week_start_target"], p_cal, label="보정(isotonic) 확률", alpha=0.85)
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    plt.legend()
    plt.title("검증 구간 — 다음 주 관심 이상 확률 (주 단위)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "timeseries_prob_alert.png", dpi=150)
    plt.close()

    summary = {
        "source": str(SAY_CSV),
        "입력_데이터": "finaldata_say.csv 만 사용 (원본 finaldata.csv 는 이 스크립트에서 읽지 않음)",
        "weekly_supervised_rows": len(tbl),
        "holdout_ratio": 0.15,
        "feature_count": len(feat_all),
        "metrics": metrics,
    }
    (OUT_DIR / "weekly_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2)[:2000])
    print(f"산출물: {OUT_DIR}/")


if __name__ == "__main__":
    main()
