#!/usr/bin/env python3
"""
과제 보강 툴킷 — 아래 순서로 실행해 outputs/assignment_toolkit/ 에 산출.

1) 데이터 프로파일 (ydata-profiling, 실패 시 간이 HTML)
2) statsmodels 지수평활 주간 total_cyano_max 베이스라인 vs naive
3) LightGBM 분위수 회귀(0.1/0.5/0.9) 구간 폭·검증 구간 커버리지
4) PDP 2변수 (sklearn + LGB 회귀)
5) 운영 비용 가중 임계값 표 (FP/FN + 가중치)
6) PuLP 선형계획 예시 (모니터링·조치 비용 스텁)
7) Plotly 주간 확률·세포 HTML (검증 구간 재계산)
8) SHAP 상호작용 상위 쌍 JSON (회귀 LGB 소표본)

실행: python3 pipeline_assignment_toolkit.py  (또는 /usr/bin/python3.10)
  - Python 3.14: pip install 시 ydata-profiling 은 제외됨 → ①단계는 간이 프로파일 HTML만 생성
"""
from __future__ import annotations

import json
import warnings
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error

try:
    from sklearn.metrics import mean_pinball_loss
except ImportError:
    mean_pinball_loss = None  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pipeline_weekly_say as wsp
from paths import OUT_ASSIGNMENT_TOOLKIT as OUT, REPO, SAY_CSV
from plot_config import setup_korean_matplotlib


def _ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("profile", "baselines", "quantile", "pdp", "cost", "optimization", "plotly", "shap_interaction"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def step1_profile() -> dict:
    """① 데이터 프로파일 HTML."""
    say = SAY_CSV
    df = pd.read_csv(say, encoding="utf-8-sig", nrows=None)
    meta: dict = {"rows": int(len(df)), "cols": int(len(df.columns))}
    prof_dir = OUT / "profile"
    html_path = prof_dir / "say_profile.html"
    try:
        from ydata_profiling import ProfileReport

        rep = ProfileReport(df, title="finaldata_say 프로파일", minimal=True)
        rep.to_file(html_path)
        meta["engine"] = "ydata-profiling"
    except Exception as e:
        meta["engine"] = "fallback"
        meta["ydata_error"] = str(e)
        desc = df.describe(include="all").to_html()
        miss = (df.isna().mean().sort_values(ascending=False).head(40)).to_frame("missing_rate").to_html()
        html_path.write_text(
            f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>say 간이 프로파일</title></head>
            <body><h1>finaldata_say (간이 — ydata-profiling 미설치/실패)</h1>
            <h2>describe</h2>{desc}<h2>결측 상위</h2>{miss}</body></html>""",
            encoding="utf-8",
        )
    meta["html"] = str(html_path.relative_to(REPO))
    (prof_dir / "profile_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def step2_statsmodels_baseline() -> dict:
    """② 주간 total_cyano_max — 지수평활 vs naive (지점별 홀드아웃)."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        meta = {"skipped": True, "reason": "statsmodels 미설치: pip install statsmodels"}
        (OUT / "baselines" / "weekly_cyano_statsmodels.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta

    df = wsp._read_say()
    w = wsp.daily_to_weekly(df)
    if "total_cyano_max" not in w.columns:
        return {"skipped": True, "reason": "total_cyano_max 없음"}

    rows = []
    for site, sub in w.groupby("채수위치"):
        sub = sub.sort_values("week_start").reset_index(drop=True)
        y = sub["total_cyano_max"].astype(float).clip(lower=0).values
        if len(y) < 12:
            continue
        cut = int(len(y) * 0.85)
        y_tr, y_te = y[:cut].astype(float), y[cut:].astype(float)
        y_tr = np.nan_to_num(y_tr, nan=np.nanmedian(y_tr[np.isfinite(y_tr)]) if np.any(np.isfinite(y_tr)) else 0.0)
        y_te = np.nan_to_num(y_te, nan=np.nanmedian(y_tr) if np.any(np.isfinite(y_tr)) else 0.0)
        # naive: 다음 주 = 직전 주
        naive_pred = np.roll(y_te, 1)
        naive_pred[0] = float(y_tr[-1])
        naive_pred = np.nan_to_num(naive_pred, nan=float(y_tr[-1]))
        m_naive = float(np.mean(np.abs(y_te - naive_pred)))

        # 단순 지수평활 1스텝 예측: 학습 구간 끝 모델로 테스트 각 스텝 1기보 (재적합 없이 근사: 전체 train ES 후 forecast len(test))
        try:
            es = ExponentialSmoothing(
                y_tr,
                trend=None,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = es.forecast(steps=len(y_te))
            fc = np.maximum(np.asarray(fc, dtype=float), 0.0)
            fc = np.nan_to_num(fc, nan=np.nanmedian(y_tr))
            m_es = float(np.mean(np.abs(y_te - fc)))
        except Exception as e:
            m_es = None
            err = str(e)
        else:
            err = None

        rows.append(
            {
                "채수위치": str(site),
                "n": int(len(y)),
                "holdout_n": int(len(y_te)),
                "mae_naive_persist": m_naive,
                "mae_exp_smooth_1step": m_es,
                "exp_smooth_error": err,
            }
        )

    out = {"by_site": rows, "note": "naive는 검증 구간에서 직전 값으로 다음 주를 맞춘 주간 persistence MAE"}
    (OUT / "baselines" / "weekly_cyano_statsmodels.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(OUT / "baselines" / "weekly_cyano_statsmodels.csv", index=False, encoding="utf-8-sig")
    return out


def _load_supervised_xy():
    df = wsp._read_say()
    tbl = wsp.build_weekly_supervised(df).dropna(subset=["y_cyano_max_next", "y_stage_max_next"]).reset_index(drop=True)
    feats = wsp.feature_cols(tbl)
    tbl[feats] = tbl[feats].replace([np.inf, -np.inf], np.nan).fillna(tbl[feats].median(numeric_only=True))
    le = pd.factorize(tbl["채수위치"])[0]
    tbl["site_code"] = le
    feat_all = feats + ["site_code"]
    X = tbl[feat_all].values
    y_reg = np.log1p(np.maximum(tbl["y_cyano_max_next"].values, 0.0))
    y_bin = tbl["y_alert_ge1_next"].values
    dates = tbl["week_start_target"]
    tr, te = wsp.time_mask(dates, 0.15)
    return tbl, feat_all, X, y_reg, y_bin, tr, te


def step3_quantile_lgb() -> dict:
    """③ 분위수 LGB 회귀."""
    from lightgbm import LGBMRegressor

    tbl, feat_all, X, y_reg, _, tr, te = _load_supervised_xy()
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_reg[tr], y_reg[te]
    alphas = [0.1, 0.5, 0.9]
    preds: dict[float, np.ndarray] = {}
    pinballs: dict[str, float] = {}
    for a in alphas:
        m = LGBMRegressor(
            objective="quantile",
            alpha=a,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=48,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds[a] = p
        if mean_pinball_loss is not None:
            pinballs[f"alpha_{a}"] = float(mean_pinball_loss(y_te, p, alpha=a))
        else:
            e = y_te - p
            pinballs[f"alpha_{a}"] = float(np.mean(np.maximum(a * e, (a - 1) * e)))

    lo, hi = preds[0.1], preds[0.9]
    y_true = y_te
    coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
    width_mean = float(np.mean(hi - lo))
    res = {
        "pinball_holdout": pinballs,
        "interval_10_90_coverage": coverage,
        "interval_10_90_mean_width_log": width_mean,
        "median_mae_log": float(mean_absolute_error(y_true, preds[0.5])),
        "median_mae_cells_approx": float(
            mean_absolute_error(np.expm1(y_true), np.maximum(np.expm1(preds[0.5]), 0.0))
        ),
    }
    (OUT / "quantile" / "quantile_lgb_summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    qdf = pd.DataFrame({"y_true_log": y_te, "q10": preds[0.1], "q50": preds[0.5], "q90": preds[0.9]})
    qdf.to_csv(OUT / "quantile" / "quantile_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    return res


def step4_pdp() -> dict:
    """④ PDP 2변수."""
    from lightgbm import LGBMRegressor

    setup_korean_matplotlib()
    tbl, feat_all, X, y_reg, _, tr, te = _load_supervised_xy()
    X_tr, y_tr = X[tr], y_reg[tr]
    m = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=40,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    m.fit(X_tr, y_tr)
    imp = m.feature_importances_
    top2 = np.argsort(imp)[::-1][:2].tolist()
    features_idx = tuple(top2)
    fig, ax = plt.subplots(figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(
        m,
        X_tr,
        features=features_idx,
        feature_names=feat_all,
        ax=ax,
        grid_resolution=20,
    )
    plt.suptitle("다음 주 log1p(세포수) — PDP (상위 특성 2개)")
    plt.tight_layout()
    pdp_path = OUT / "pdp" / "pdp_top2_lgb_regression.png"
    plt.savefig(pdp_path, dpi=150)
    plt.close()
    meta = {
        "features": [feat_all[i] for i in top2],
        "png": str(pdp_path.relative_to(REPO)),
    }
    (OUT / "pdp" / "pdp_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def step5_cost_table() -> dict:
    """⑤ FP/FN 가중 비용 + 임계값 스윕."""
    from lightgbm import LGBMClassifier

    tbl, feat_all, X, _, y_bin, tr, te = _load_supervised_xy()
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_bin[tr], y_bin[te]
    m = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)[:, 1]
    w_fp, w_fn = 1.0, 3.0  # 미발령을 관심으로 잘못 보낸 비용 1, 놓친 비용 3 (예시 가중)
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (proba >= t).astype(int)
        fp = int(np.sum((pred == 1) & (y_te == 0)))
        fn = int(np.sum((pred == 0) & (y_te == 1)))
        tp = int(np.sum((pred == 1) & (y_te == 1)))
        tn = int(np.sum((pred == 0) & (y_te == 0)))
        rows.append(
            {
                "threshold": round(float(t), 3),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                "weighted_cost_fp_fn": float(w_fp * fp + w_fn * fn),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost" / "threshold_cost_operational.csv", index=False, encoding="utf-8-sig")
    meta = {"weights": {"false_positive": w_fp, "false_negative": w_fn}, "csv": "cost/threshold_cost_operational.csv"}
    (OUT / "cost" / "cost_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["threshold"], df["recall"], label="재현율")
    ax.plot(df["threshold"], df["precision"], label="정밀도")
    ax2 = ax.twinx()
    ax2.plot(df["threshold"], df["weighted_cost_fp_fn"], color="tab:red", alpha=0.7, label="가중비용(FP+FN)")
    ax.set_xlabel("임계값")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("이진 LGB — 임계값 vs 재현율·정밀도·가중비용(예시)")
    plt.tight_layout()
    plt.savefig(OUT / "cost" / "threshold_cost_curve.png", dpi=150)
    plt.close()
    return {
        "csv": "cost/threshold_cost_operational.csv",
        "png": "cost/threshold_cost_curve.png",
        "weights": meta["weights"],
    }


def step6_pulp_example() -> dict:
    """⑥ PuLP 예시: 모니터링·조치 이진 선택 (교육용 스텁)."""
    try:
        import pulp
    except ImportError:
        meta = {"skipped": True, "reason": "pulp 미설치: pip install pulp"}
        (OUT / "optimization" / "pulp_example.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    prob = pulp.LpProblem("AlgaeResponse", pulp.LpMinimize)
    x_m = pulp.LpVariable("monitor_intensive", cat="Binary")
    x_a = pulp.LpVariable("treatment_active", cat="Binary")
    prob += 10 * x_m + 50 * x_a, "TotalCost"
    prob += x_m + x_a >= 1, "AtLeastOneMeasure"
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = {
        "status": pulp.LpStatus[prob.status],
        "monitor_intensive": int(pulp.value(x_m) or 0),
        "treatment_active": int(pulp.value(x_a) or 0),
        "objective_cost": float(pulp.value(prob.objective) or 0),
        "note": "실제 과제에서는 예측 확률·유입·단계에 따라 제약을 두고 선형화한 뒤 LP/MIP로 확장 가능",
    }
    (OUT / "optimization" / "pulp_example.json").write_text(json.dumps(sol, ensure_ascii=False, indent=2), encoding="utf-8")
    return sol


def step7_plotly_html() -> dict:
    """⑦ Plotly — 검증 구간 관심이상 확률·예측 세포 (주간)."""
    try:
        import plotly.express as px
    except ImportError:
        meta = {"skipped": True, "reason": "plotly 미설치"}
        (OUT / "plotly" / "plotly_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    from lightgbm import LGBMClassifier, LGBMRegressor

    tbl, feat_all, X, y_reg, y_bin, tr, te = _load_supervised_xy()
    X_tr, X_te = X[tr], X[te]
    meta_te = tbl.loc[te].sort_values(["week_start_target", "채수위치"]).reset_index(drop=True)

    bm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    bm.fit(X_tr, y_bin[tr])
    p_alert = bm.predict_proba(X_te)[:, 1]

    reg = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=40,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    reg.fit(X_tr, y_reg[tr])
    pred_log = reg.predict(X_te)

    plot_df = meta_te[["week_start_target", "채수위치"]].copy()
    plot_df["p_alert_lgb"] = p_alert
    plot_df["pred_cyano_max_approx"] = np.expm1(np.maximum(pred_log, 0))
    plot_df["week_start_target"] = pd.to_datetime(plot_df["week_start_target"])

    fig = px.line(
        plot_df,
        x="week_start_target",
        y="p_alert_lgb",
        color="채수위치",
        title="검증 구간 — 다음 주 관심 이상 확률 (LGB, 주간)",
    )
    path1 = OUT / "plotly" / "holdout_prob_alert.html"
    fig.write_html(path1, include_plotlyjs="cdn")

    fig2 = px.line(
        plot_df,
        x="week_start_target",
        y="pred_cyano_max_approx",
        color="채수위치",
        title="검증 구간 — 예측 주간 max 세포수 (근사)",
    )
    path2 = OUT / "plotly" / "holdout_pred_cyano.html"
    fig2.write_html(path2, include_plotlyjs="cdn")

    meta = {"prob_html": str(path1.relative_to(REPO)), "cyano_html": str(path2.relative_to(REPO))}
    (OUT / "plotly" / "plotly_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def step8_shap_interaction_top_pairs() -> dict:
    """⑧ SHAP 상호작용 (회귀 LGB, 홀드아웃 소표본) — 상위 절대값 쌍."""
    try:
        import shap
        from lightgbm import LGBMRegressor
    except ImportError as e:
        return {"skipped": True, "reason": str(e)}

    tbl, feat_all, X, y_reg, _, tr, te = _load_supervised_xy()
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_reg[tr], y_reg[te]
    m = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.06,
        num_leaves=40,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    m.fit(X_tr, y_tr)
    n = min(120, len(X_te))
    Xs = X_te[:n]
    explainer = shap.TreeExplainer(m)
    iv = explainer.shap_interaction_values(Xs)
    if isinstance(iv, list):
        iv = np.stack(iv, axis=0)
    mat = np.mean(np.abs(iv), axis=0)
    F = mat.shape[0]
    pairs: list[tuple[str, str, float]] = []
    for i in range(F):
        for j in range(i + 1, F):
            pairs.append((feat_all[i], feat_all[j], float(mat[i, j])))
    pairs.sort(key=lambda x: -x[2])
    top = [{"f1": a, "f2": b, "mean_abs_interaction": c} for a, b, c in pairs[:12]]
    out = {"n_sample": n, "top_pairs": top}
    (OUT / "shap_interaction" / "shap_interaction_top_pairs.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "shap_interaction" / "shap_interaction_top_pairs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def _safe(name: str, fn, log: dict) -> None:
    try:
        log[name] = fn()
    except Exception as e:
        log[name] = {"error": str(e)}


def main() -> None:
    setup_korean_matplotlib()
    _ensure_out()
    log: dict[str, object] = {}

    print("① 프로파일 …")
    _safe("step1_profile", step1_profile, log)
    print("② statsmodels 베이스라인 …")
    _safe("step2_baselines", step2_statsmodels_baseline, log)
    print("③ 분위수 LGB …")
    _safe("step3_quantile", step3_quantile_lgb, log)
    print("④ PDP …")
    _safe("step4_pdp", step4_pdp, log)
    print("⑤ 비용 임계값 …")
    _safe("step5_cost", step5_cost_table, log)
    print("⑥ PuLP …")
    _safe("step6_pulp", step6_pulp_example, log)
    print("⑦ Plotly …")
    _safe("step7_plotly", step7_plotly_html, log)
    print("⑧ SHAP 상호작용 …")
    _safe("step8_shap_interaction", step8_shap_interaction_top_pairs, log)

    (OUT / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"완료: {OUT}/")


if __name__ == "__main__":
    main()
