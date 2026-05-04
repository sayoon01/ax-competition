#!/usr/bin/env python3
"""
일 단위 D+7 발령단계 예측 (산출물: outputs/daily_d7/)

- **입력은 finaldata_say.csv 만** 사용합니다. (원본 finaldata.csv 병합·라벨·roll7 등은 노트북/전처리에서 say 생성)
- LightGBM / XGBoost, SHAP, 시계열 검증 그림
"""
from __future__ import annotations

import json
import warnings
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from paths import OUT_DAILY_D7 as OUT_DIR, SAY_CSV
from plot_config import setup_korean_matplotlib

warnings.filterwarnings("ignore", category=UserWarning)

setup_korean_matplotlib()

# say에 반드시 있어야 하는 열 (없으면 전처리로 say를 먼저 생성)
_SAY_REQUIRED_COLS = frozenset(
    {
        "조사일",
        "채수위치",
        "발령단계",
        "발령단계_코드",
        "target_발령단계_D7_코드",
        "roll7_mean_수온(℃)",
        "채수위치_코드",
    }
)

# 발령단계 ↔ 정수 (데이터에 등장하는 값 기준, 대발생은 예비)
STAGE_ORDER = ["미발령", "관심", "경계", "조류대발생"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_ORDER)}
INT_TO_STAGE = {i: s for s, i in STAGE_TO_INT.items()}

THRESHOLDS = [
    (1_000, "관심"),
    (10_000, "경계"),
    (1_000_000, "조류대발생"),
]


def stage_from_cells(total_cyano: float) -> int:
    if pd.isna(total_cyano):
        return 0
    v = float(total_cyano)
    if v >= 1_000_000:
        return 3
    if v >= 10_000:
        return 2
    if v >= 1_000:
        return 1
    return 0


def map_발령단계(s: object) -> int:
    if pd.isna(s):
        return 0
    t = str(s).strip()
    if t in STAGE_TO_INT:
        return STAGE_TO_INT[t]
    if "대발생" in t or "조류대발생" in t:
        return 3
    return 0


def enrich_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["조사일"] = pd.to_datetime(out["조사일"], errors="coerce")
    out = out.sort_values(["채수위치", "조사일"]).reset_index(drop=True)

    out["발령단계_코드"] = out["발령단계"].map(map_발령단계)
    out["세포수기준_단계_코드"] = out["total_cyano"].map(stage_from_cells)
    out["발령단계_vs_세포수_일치"] = out["발령단계_코드"] == out["세포수기준_단계_코드"]

    out["초과_관심"] = (out["total_cyano"] >= 1_000).fillna(False)
    out["초과_경계"] = (out["total_cyano"] >= 10_000).fillna(False)
    out["초과_대발생"] = (out["total_cyano"] >= 1_000_000).fillna(False)

    g = out.groupby("채수위치", group_keys=False)

    def consec_two(col: str) -> pd.Series:
        def inner(s: pd.Series) -> pd.Series:
            cur = s.fillna(False).astype(np.bool_)
            pr = s.shift(1)
            prev = pr.where(pr.notna(), False).astype(np.bool_)
            return cur & prev

        return g[col].transform(inner)

    out["연속2일_관심기준"] = consec_two("초과_관심")
    out["연속2일_경계기준"] = consec_two("초과_경계")
    out["연속2일_대발생기준"] = consec_two("초과_대발생")

    # D+7 동일 지점 발령단계 (예측 정답 후보)
    out["target_발령단계_D7_코드"] = g["발령단계_코드"].transform(lambda s: s.shift(-7))

    # 차주(다음 7일, t+1..t+7) 최고 단계 — 보수적 시나리오
    out["target_발령단계_차주7일_최고단계_코드"] = g["발령단계_코드"].transform(
        lambda s: s.shift(-1).rolling(window=7, min_periods=1).max()
    )

    return out


def load_prepared_say() -> pd.DataFrame:
    """finaldata_say.csv 로드·검증. 이 파일은 전처리 단계에서만 갱신."""
    if not SAY_CSV.exists():
        raise FileNotFoundError(
            f"{SAY_CSV} 가 없습니다. data_preprocessing 등으로 say 파일을 만든 뒤 실행하세요."
        )
    df = pd.read_csv(SAY_CSV, encoding="utf-8-sig")
    miss = _SAY_REQUIRED_COLS - set(df.columns)
    if miss:
        raise ValueError(f"finaldata_say.csv 에 필요한 열이 없습니다: {sorted(miss)}")
    df = df.copy()
    df["조사일"] = pd.to_datetime(df["조사일"], errors="coerce")
    return df.sort_values(["채수위치", "조사일"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("채수위치", group_keys=False)

    def roll_mean(col: str) -> pd.Series:
        return g[col].transform(lambda s: s.rolling(7, min_periods=1).mean())

    def roll_sum(col: str) -> pd.Series:
        return g[col].transform(lambda s: s.rolling(7, min_periods=1).sum())

    num_cols = [
        "수온(℃)",
        "Chl-a (㎎/㎥)",
        "total_cyano",
        "pH",
        "DO(㎎/L)",
        "탁도",
        "투명도",
        "유입량(㎥/s)",
        "총방류량(㎥/s)",
        "합계 일조시간(hr)",
        "일강수량(mm)",
        "강우량(mm)",
    ]
    for c in num_cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[f"roll7_mean_{c}"] = roll_mean(c)

    out["일일강수합_mm"] = out["일강수량(mm)"].fillna(0)
    if "강우량(mm)" in out.columns:
        out["일일강수합_mm"] = out["일일강수합_mm"] + out["강우량(mm)"].fillna(0)
    out["roll7_누적강수_mm"] = g["일일강수합_mm"].transform(lambda s: s.rolling(7, min_periods=1).sum())

    out["유입_방류_차"] = (out["유입량(㎥/s)"] - out["총방류량(㎥/s)"]).astype(float)
    out["roll7_mean_유입_방류_차"] = g["유입_방류_차"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    # 전주 대비 (직전 7일 평균 vs 그 이전 7일 평균)
    def wow_delta(col: str) -> pd.Series:
        def inner(s: pd.Series) -> pd.Series:
            cur = s.rolling(7, min_periods=1).mean()
            prev = cur.shift(7)
            return cur - prev

        return g[col].transform(inner)

    for c in ["수온(℃)", "Chl-a (㎎/㎥)", "total_cyano"]:
        if c in out.columns:
            out[f"wow_delta_{c}"] = wow_delta(c)

    def rising3(tc: pd.Series) -> pd.Series:
        d1 = tc.diff(1) > 0
        d2 = tc.diff(1).shift(1) > 0
        d3 = tc.diff(1).shift(2) > 0
        return (d1 & d2 & d3).astype(int)

    out["연속3일_세포수_증가"] = g["total_cyano"].transform(rising3)

    out["월"] = out["조사일"].dt.month
    out["연중일"] = out["조사일"].dt.dayofyear
    out["계절"] = out["월"].map(
        lambda m: 0 if m in (12, 1, 2) else 1 if m in (3, 4, 5) else 2 if m in (6, 7, 8) else 3
    )
    out["여름철_6_8월"] = out["월"].between(6, 8).astype(int)

    le = LabelEncoder()
    out["채수위치_코드"] = le.fit_transform(out["채수위치"].astype(str))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "조사일",
        "채수위치",
        "발령단계",
        "발령단계_코드",
        "세포수기준_단계_코드",
        "발령단계_vs_세포수_일치",
        "target_발령단계_D7_코드",
        "target_발령단계_차주7일_최고단계_코드",
    }
    feats: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            feats.append(c)
    return sorted(set(feats))


def time_holdout_mask(dates: pd.Series, holdout_ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    u = np.sort(dates.dropna().unique())
    cut_idx = int(len(u) * (1.0 - holdout_ratio))
    cut_date = u[max(cut_idx, 1) - 1]
    tr = dates <= cut_date
    te = dates > cut_date
    return tr.values, te.values


def train_eval_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: list[str],
    class_labels: list[str],
) -> dict:
    metrics: dict = {}

    try:
        from lightgbm import LGBMClassifier

        lgb = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=48,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        lgb.fit(X_train, y_train)
        pred = lgb.predict(X_test)
        metrics["lightgbm"] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "report": classification_report(
                y_test, pred, labels=list(range(len(class_labels))), target_names=class_labels, zero_division=0
            ),
        }
        model_lgb = lgb
    except Exception as e:
        metrics["lightgbm"] = {"error": str(e)}
        model_lgb = None

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=0,
            objective="multi:softprob",
            num_class=len(class_labels),
            eval_metric="mlogloss",
        )
        xgb.fit(X_train, y_train)
        pred_x = xgb.predict(X_test)
        metrics["xgboost"] = {
            "accuracy": float(accuracy_score(y_test, pred_x)),
            "f1_macro": float(f1_score(y_test, pred_x, average="macro", zero_division=0)),
            "report": classification_report(
                y_test, pred_x, labels=list(range(len(class_labels))), target_names=class_labels, zero_division=0
            ),
        }
        model_xgb = xgb
    except Exception as e:
        metrics["xgboost"] = {"error": str(e)}
        model_xgb = None

    # SHAP (LightGBM, 다중분류: 클래스별 |SHAP| 평균 후 특성 중요도 막대)
    if model_lgb is not None:
        try:
            import shap

            setup_korean_matplotlib()
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            n_shap = min(400, len(X_test))
            X_shap = X_test[:n_shap]
            explainer = shap.TreeExplainer(model_lgb)
            shap_vals = explainer.shap_values(X_shap)
            if isinstance(shap_vals, list):
                stacked = np.mean(np.stack([np.abs(s) for s in shap_vals], axis=0), axis=0)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                stacked = np.abs(shap_vals).mean(axis=2)
            else:
                stacked = np.abs(np.asarray(shap_vals))
            arr = stacked.mean(axis=0)
            order = np.argsort(arr)[::-1][:30]
            plt.figure(figsize=(10, 8))
            plt.barh(np.array(feat_names)[order][::-1], arr[order][::-1])
            plt.xlabel("클래스 평균 mean |SHAP|")
            plt.title("LightGBM — 특성 중요도 (검증 표본)")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "shap_mean_abs_bar.png", dpi=150)
            plt.close()

            k = int(np.bincount(y_train.astype(int)).argmax())
            k = min(k, len(class_labels) - 1)
            if isinstance(shap_vals, list) and len(shap_vals) > k:
                sv_plot = shap_vals[k]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                sv_plot = shap_vals[:, :, k]
            else:
                sv_plot = None
            if sv_plot is not None:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    sv_plot,
                    X_shap,
                    feature_names=feat_names,
                    plot_type="dot",
                    max_display=18,
                    show=False,
                )
                plt.title(f"SHAP (LightGBM, 클래스={class_labels[k]})")
                plt.tight_layout()
                plt.savefig(OUT_DIR / "shap_beeswarm_class.png", dpi=150)
                plt.close()
        except Exception as e:
            metrics["shap_error"] = str(e)

    return metrics, model_lgb, model_xgb


def plot_timeseries(
    dates: pd.Series,
    sites: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """지점별로 검증 구간 실제/예측 단계 시계열"""
    setup_korean_matplotlib()
    df = pd.DataFrame({"d": dates, "site": sites, "y": y_true, "p": y_pred})
    sites_u = df["site"].unique()
    n = len(sites_u)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, sites_u):
        sub = df[df["site"] == s].sort_values("d")
        ax.plot(sub["d"], sub["y"], label="실제(코드)", alpha=0.8)
        ax.plot(sub["d"], sub["p"], "--", label="예측(코드)", alpha=0.8)
        ax.set_ylabel("단계 코드")
        ax.legend(loc="upper right")
        ax.set_title(str(s))
    fig.suptitle(title)
    plt.xlabel("조사일")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    enriched = load_prepared_say()

    # --- 학습 테이블: D+7 단계 예측 (주 타깃)
    target_col = "target_발령단계_D7_코드"
    dfm = enriched.dropna(subset=[target_col]).copy()
    feats = feature_columns(dfm)
    dfm[feats] = dfm[feats].replace([np.inf, -np.inf], np.nan)
    dfm[feats] = dfm[feats].fillna(dfm[feats].median(numeric_only=True))

    X = dfm[feats].values
    y = dfm[target_col].astype(int).values

    present = sorted(set(y.tolist()))
    class_labels = [INT_TO_STAGE[i] for i in range(max(present) + 1)]
    # 라벨이 0,1,2만 있으면 num_class=3
    n_class = int(y.max()) + 1
    y = np.clip(y, 0, n_class - 1)
    class_labels = [INT_TO_STAGE[i] for i in range(n_class)]

    dates = dfm["조사일"]
    sites = dfm["채수위치"]
    tr_m, te_m = time_holdout_mask(dates, 0.15)

    X_train, X_test = X[tr_m], X[te_m]
    y_train, y_test = y[tr_m], y[te_m]
    d_train, d_test = dates[tr_m].values, dates[te_m].values
    s_train, s_test = sites[tr_m].values, sites[te_m].values

    metrics, lgb_model, xgb_model = train_eval_models(
        X_train, y_train, X_test, y_test, feats, class_labels
    )

    pred_lgb_te: np.ndarray | None = None
    pred_xgb_te: np.ndarray | None = None
    if lgb_model is not None:
        pred_lgb_te = lgb_model.predict(X_test)
        plot_timeseries(
            pd.to_datetime(d_test),
            pd.Series(s_test),
            y_test,
            pred_lgb_te,
            "검증 구간: 발령단계 코드 (실제 vs LightGBM 예측, D+7)",
            OUT_DIR / "timeseries_holdout_lightgbm.png",
        )
    if xgb_model is not None:
        pred_xgb_te = xgb_model.predict(X_test)

    # 검증 구간 행 단위 예측 (제출·보고용)
    hold = dfm.loc[te_m, ["조사일", "채수위치", "발령단계", "발령단계_코드"]].copy()
    hold["y_target_D7_stage_code"] = y_test
    hold["y_target_D7_stage_name"] = [INT_TO_STAGE.get(int(c), str(c)) for c in y_test]
    if pred_lgb_te is not None:
        hold["pred_lightgbm_stage_code"] = pred_lgb_te.astype(int)
        hold["pred_lightgbm_stage_name"] = [INT_TO_STAGE.get(int(c), str(c)) for c in pred_lgb_te.astype(int)]
    if pred_xgb_te is not None:
        hold["pred_xgboost_stage_code"] = pred_xgb_te.astype(int)
        hold["pred_xgboost_stage_name"] = [INT_TO_STAGE.get(int(c), str(c)) for c in pred_xgb_te.astype(int)]
    hold["holdout_note"] = "시간 홀드아웃 후반 약 15% (고유 조사일 기준)"
    hold.to_csv(OUT_DIR / "holdout_predictions_daily_d7.csv", index=False, encoding="utf-8-sig")

    # 요약 저장
    summary = {
        "input_csv": str(SAY_CSV),
        "input_rows": len(enriched),
        "modeling_rows": len(dfm),
        "target": target_col,
        "holdout_ratio": 0.15,
        "features_n": len(feats),
        "classes": class_labels,
        "metrics": metrics,
        "발령단계_vs_세포수_일치율": float(enriched["발령단계_vs_세포수_일치"].mean()),
    }
    (OUT_DIR / "pipeline_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, ensure_ascii=False, indent=2))
    for name, m in metrics.items():
        if "accuracy" in m:
            print(f"{name}: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")
    print(f"입력: {SAY_CSV} (변경 없음)")
    print(f"산출물: {OUT_DIR}/")


if __name__ == "__main__":
    main()
