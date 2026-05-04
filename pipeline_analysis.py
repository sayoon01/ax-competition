#!/usr/bin/env python3
"""
finaldata_say 기반 확장 진단 (홀드아웃 15% = pipeline_weekly_say와 동일 분할).

산출: outputs/analysis/
  - analysis_summary.json
  - confusion_matrix_stage.csv / .png
  - metrics_by_site.csv, metrics_by_season.csv
  - threshold_binary_lgb.csv (+ PNG)
  - lookback_ablation.json
  - data_quality_daily.json (+ 선택 PNG)
  - persistence_vs_model.json

실행: python3 pipeline_analysis.py (matplotlib/lightgbm 필요; 예: /usr/bin/python3.10)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

import pipeline_weekly_say as wsp
from paths import OUT_ANALYSIS as OUT
from plot_config import setup_korean_matplotlib

warnings.filterwarnings("ignore", category=UserWarning)


def _stage_from_cyano(v: float) -> int:
    if pd.isna(v):
        return 0
    x = float(v)
    if x >= 1_000_000:
        return 3
    if x >= 10_000:
        return 2
    if x >= 1_000:
        return 1
    return 0


def build_supervised_lookback(df: pd.DataFrame, L: int) -> pd.DataFrame:
    """L주 메인 창 (lw2/lw1은 weekly와 동일). L==4 이면 weekly_say 원본과 동일 표."""
    if L == 4:
        return wsp.build_weekly_supervised(df)
    weekly = wsp.daily_to_weekly(df)
    all_rows: list[dict] = []
    for site, sub in weekly.groupby("채수위치"):
        sub = sub.sort_values("week_start").reset_index(drop=True)
        cols_track = [c for c in sub.columns if c not in ("채수위치", "week", "week_start")]
        start_k = max(L, 2)
        for k in range(start_k, len(sub)):
            past = sub.iloc[k - L : k]
            nxt = sub.iloc[k]
            row: dict = {
                "채수위치": str(site),
                "week_start_target": nxt["week_start"],
            }
            for c in cols_track:
                s = pd.to_numeric(past[c], errors="coerce")
                row[f"lb{L}_mean_{c}"] = float(s.mean())
                row[f"lb{L}_std_{c}"] = float(s.std(ddof=0)) if len(s) > 1 else 0.0
                row[f"lb{L}_slope_{c}"] = float(s.iloc[-1] - s.iloc[0])

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


def feature_cols_for_L(tbl: pd.DataFrame, L: int) -> list[str]:
    if L == 4:
        return wsp.feature_cols(tbl)
    return sorted(
        [c for c in tbl.columns if c.startswith(f"lb{L}_") or c.startswith("lw2_") or c.startswith("lw1_")]
    )


def enrich_persistence(tbl: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """직전 주 발령단계·max 세포수 (weekly 행과 동일 키로 merge)."""
    w = wsp.daily_to_weekly(df)
    rows: list[dict] = []
    for site, sub in w.groupby("채수위치"):
        sub = sub.sort_values("week_start").reset_index(drop=True)
        for k in range(4, len(sub)):
            nxt = sub.iloc[k]
            prev = sub.iloc[k - 1]
            rows.append(
                {
                    "채수위치": str(site),
                    "week_start_target": pd.Timestamp(nxt["week_start"]),
                    "persist_stage": int(float(prev["발령단계_코드_max"]))
                    if "발령단계_코드_max" in prev
                    else 0,
                    "persist_cyano_max": float(prev["total_cyano_max"])
                    if "total_cyano_max" in prev and pd.notna(prev["total_cyano_max"])
                    else np.nan,
                }
            )
    p = pd.DataFrame(rows)
    out = tbl.merge(p, on=["채수위치", "week_start_target"], how="left")
    return out


def daily_data_quality(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["조사일"] = pd.to_datetime(d["조사일"], errors="coerce")
    d = d.dropna(subset=["조사일", "채수위치"])
    gaps: dict[str, float] = {}
    for site, g in d.groupby("채수위치"):
        g = g.sort_values("조사일")
        dd = g["조사일"].diff().dt.days.dropna()
        gaps[str(site)] = float(dd.median()) if len(dd) else float("nan")

    miss = d.isna().mean().sort_values(ascending=False)
    miss_top = {str(k): round(float(v), 4) for k, v in miss.head(25).items()}

    mismatch_rate = None
    if "발령단계_코드" in d.columns and "total_cyano" in d.columns:
        cy = d["total_cyano"].map(_stage_from_cyano)
        br = pd.to_numeric(d["발령단계_코드"], errors="coerce").fillna(0).astype(int)
        mismatch_rate = float((cy != br).mean())

    return {
        "n_rows": int(len(d)),
        "median_gap_days_by_site": gaps,
        "missing_rate_top25": miss_top,
        "발령단계코드_vs_세포수임계단계_불일치율": mismatch_rate,
    }


def train_lgb_stage(X_tr, y_tr, X_te, y_te) -> tuple[np.ndarray, object]:
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=48,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return pred, clf


def train_lgb_reg(X_tr, y_tr, X_te) -> np.ndarray:
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
    reg.fit(X_tr, y_tr)
    return reg.predict(X_te)


def train_lgb_bin(X_tr, y_tr, X_te) -> tuple[np.ndarray, np.ndarray]:
    from lightgbm import LGBMClassifier

    m = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path) -> None:
    setup_korean_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("실제")
    ax.set_xlabel("예측")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.title("홀드아웃 — 발령단계 혼동행렬 (LGB)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    setup_korean_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)

    df = wsp._read_say()
    dq = daily_data_quality(df)
    (OUT / "data_quality_daily.json").write_text(json.dumps(dq, ensure_ascii=False, indent=2), encoding="utf-8")

    tbl = wsp.build_weekly_supervised(df)
    tbl = tbl.dropna(subset=["y_cyano_max_next", "y_stage_max_next"]).reset_index(drop=True)
    tbl = enrich_persistence(tbl, df)

    feats = wsp.feature_cols(tbl)
    tbl[feats] = tbl[feats].replace([np.inf, -np.inf], np.nan).fillna(tbl[feats].median(numeric_only=True))
    le = pd.factorize(tbl["채수위치"])[0]
    tbl["site_code"] = le
    feat_all = feats + ["site_code"]
    X = tbl[feat_all].values
    y_stage = tbl["y_stage_max_next"].clip(0, 3).astype(int).values
    y_bin = tbl["y_alert_ge1_next"].values
    y_reg = np.log1p(np.maximum(tbl["y_cyano_max_next"].values, 0.0))
    dates = tbl["week_start_target"]
    tr, te = wsp.time_mask(dates, 0.15)

    X_tr, X_te = X[tr], X[te]
    y_st_tr, y_st_te = y_stage[tr], y_stage[te]
    y_bin_tr, y_bin_te = y_bin[tr], y_bin[te]
    y_reg_tr, y_reg_te = y_reg[tr], y_reg[te]
    meta_te = tbl.loc[te].reset_index(drop=True)

    pred_st, _ = train_lgb_stage(X_tr, y_st_tr, X_te, y_st_te)
    pred_reg = train_lgb_reg(X_tr, y_reg_tr, X_te)
    pred_bin, proba_bin = train_lgb_bin(X_tr, y_bin_tr, X_te)

    names = ["미발령", "관심", "경계", "대발생"]
    ncls = int(max(y_st_te.max(), pred_st.max())) + 1
    labels = names[:ncls]
    cm = confusion_matrix(y_st_te, pred_st, labels=list(range(ncls)))
    np.savetxt(OUT / "confusion_matrix_stage.csv", cm, delimiter=",", fmt="%d")
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(ncls)], columns=[f"pred_{i}" for i in range(ncls)]).to_csv(
        OUT / "confusion_matrix_stage_labeled.csv", encoding="utf-8-sig"
    )
    plot_confusion(cm, labels, OUT / "confusion_matrix_stage.png")

    # 지점별
    rows_site = []
    for site in meta_te["채수위치"].unique():
        m = meta_te["채수위치"] == site
        ix = np.where(m.values)[0]
        if len(ix) < 3:
            continue
        yt = y_st_te[ix]
        yp = pred_st[ix]
        yb = y_bin_te[ix]
        pb = proba_bin[ix]
        rows_site.append(
            {
                "채수위치": str(site),
                "n_holdout": int(len(ix)),
                "stage_accuracy": float(accuracy_score(yt, yp)),
                "stage_f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "binary_auc": float(roc_auc_score(yb, pb)) if len(np.unique(yb)) > 1 else None,
                "binary_recall_0.5": float(recall_score(yb, (pb >= 0.5).astype(int), pos_label=1, zero_division=0)),
                "reg_mae_cells": float(
                    mean_absolute_error(
                        np.expm1(y_reg_te[ix]),
                        np.maximum(np.expm1(pred_reg[ix]), 0.0),
                    )
                ),
            }
        )
    site_df = pd.DataFrame(rows_site)
    site_df.to_csv(OUT / "metrics_by_site.csv", index=False, encoding="utf-8-sig")
    if len(site_df):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ypos = np.arange(len(site_df))
        ax.barh(ypos - 0.15, site_df["stage_accuracy"], height=0.28, label="단계 accuracy")
        br = site_df["binary_recall_0.5"].fillna(0)
        ax.barh(ypos + 0.15, br, height=0.28, label="이진 recall@0.5")
        ax.set_yticks(ypos)
        ax.set_yticklabels(site_df["채수위치"])
        ax.set_xlim(0, 1.05)
        ax.legend(loc="lower right")
        ax.set_title("홀드아웃 — 지점별 지표")
        plt.tight_layout()
        plt.savefig(OUT / "metrics_by_site.png", dpi=150)
        plt.close()

    # 계절
    meta_te = meta_te.copy()
    meta_te["month"] = pd.to_datetime(meta_te["week_start_target"]).dt.month

    def season(m: int) -> str:
        if m in (12, 1, 2):
            return "겨울"
        if m in (3, 4, 5):
            return "봄"
        if m in (6, 7, 8):
            return "여름"
        return "가을"

    meta_te["season"] = meta_te["month"].map(season)
    rows_season = []
    for s in ["봄", "여름", "가을", "겨울"]:
        m = meta_te["season"] == s
        ix = np.where(m.values)[0]
        if len(ix) < 2:
            continue
        rows_season.append(
            {
                "season": s,
                "n_holdout": int(len(ix)),
                "stage_accuracy": float(accuracy_score(y_st_te[ix], pred_st[ix])),
                "stage_f1_macro": float(f1_score(y_st_te[ix], pred_st[ix], average="macro", zero_division=0)),
                "binary_recall_0.5": float(
                    recall_score(y_bin_te[ix], (proba_bin[ix] >= 0.5).astype(int), pos_label=1, zero_division=0)
                ),
            }
        )
    season_df = pd.DataFrame(rows_season)
    season_df.to_csv(OUT / "metrics_by_season.csv", index=False, encoding="utf-8-sig")
    if len(season_df):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(season_df))
        w = 0.35
        ax.bar(x - w / 2, season_df["stage_accuracy"], width=w, label="단계 accuracy")
        ax.bar(x + w / 2, season_df["binary_recall_0.5"], width=w, label="이진 recall@0.5")
        ax.set_xticks(x)
        ax.set_xticklabels(season_df["season"])
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("홀드아웃 — 계절별 지표")
        plt.tight_layout()
        plt.savefig(OUT / "metrics_by_season.png", dpi=150)
        plt.close()

    # 임계값 스윕
    thr_rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pb = (proba_bin >= t).astype(int)
        thr_rows.append(
            {
                "threshold": round(float(t), 3),
                "recall_alert": float(recall_score(y_bin_te, pb, pos_label=1, zero_division=0)),
                "precision_alert": float(precision_score(y_bin_te, pb, pos_label=1, zero_division=0)),
                "f1": float(f1_score(y_bin_te, pb, zero_division=0)),
            }
        )
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(OUT / "threshold_binary_lgb.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr_df["threshold"], thr_df["recall_alert"], label="재현율(관심↑)")
    ax.plot(thr_df["threshold"], thr_df["precision_alert"], label="정밀도(관심↑)")
    ax.plot(thr_df["threshold"], thr_df["f1"], label="F1")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("이진 확률 임계값")
    ax.legend()
    ax.set_title("LGB 이진 — 임계값에 따른 재현율·정밀도·F1 (홀드아웃)")
    plt.tight_layout()
    plt.savefig(OUT / "threshold_binary_lgb.png", dpi=150)
    plt.close()

    # persistence vs model
    persist_st = meta_te["persist_stage"].fillna(0).astype(int).values.clip(0, 3)
    persist_cy = meta_te["persist_cyano_max"].values
    mask_cy = np.isfinite(persist_cy) & np.isfinite(np.expm1(y_reg_te))
    pers_stage_acc = float(accuracy_score(y_st_te, persist_st))
    model_stage_acc = float(accuracy_score(y_st_te, pred_st))
    pers_cy_mae = (
        float(mean_absolute_error(np.expm1(y_reg_te)[mask_cy], np.maximum(persist_cy[mask_cy], 0)))
        if mask_cy.sum() > 0
        else None
    )
    model_cy_mae = float(mean_absolute_error(np.expm1(y_reg_te), np.maximum(np.expm1(pred_reg), 0.0)))
    med_c = float(np.nanmedian(persist_cy)) if np.any(np.isfinite(persist_cy)) else 0.0
    pc_log = np.log1p(np.maximum(np.nan_to_num(persist_cy, nan=med_c), 0.0))
    try:
        pers_reg_r2 = float(r2_score(y_reg_te, pc_log))
    except Exception:
        pers_reg_r2 = None

    persistence_block = {
        "stage_accuracy_persistence": pers_stage_acc,
        "stage_accuracy_lgb": model_stage_acc,
        "stage_accuracy_delta_lgb_minus_persist": round(model_stage_acc - pers_stage_acc, 4),
        "cyano_mae_cells_persistence_prev_week": pers_cy_mae,
        "cyano_mae_cells_lgb_log1p": model_cy_mae,
        "reg_r2_log1p_persistence": pers_reg_r2,
        "reg_r2_log1p_lgb": float(r2_score(y_reg_te, pred_reg)),
    }
    (OUT / "persistence_vs_model.json").write_text(
        json.dumps(persistence_block, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 룩백 3·4·5
    lookback_results: dict[str, dict] = {}
    for L in (3, 4, 5):
        tL = build_supervised_lookback(df, L).dropna(subset=["y_cyano_max_next", "y_stage_max_next"]).reset_index(
            drop=True
        )
        fL = feature_cols_for_L(tL, L)
        tL[fL] = tL[fL].replace([np.inf, -np.inf], np.nan).fillna(tL[fL].median(numeric_only=True))
        leL = pd.factorize(tL["채수위치"])[0]
        tL["site_code"] = leL
        f_all = fL + ["site_code"]
        XL = tL[f_all].values
        yL = tL["y_stage_max_next"].clip(0, 3).astype(int).values
        dL = tL["week_start_target"]
        trL, teL = wsp.time_mask(dL, 0.15)
        if teL.sum() < 10 or trL.sum() < 50:
            lookback_results[f"L{L}"] = {"skipped": True, "reason": "샘플 부족"}
            continue
        predL, _ = train_lgb_stage(XL[trL], yL[trL], XL[teL], yL[teL])
        lookback_results[f"L{L}"] = {
            "n_rows": int(len(tL)),
            "n_holdout": int(teL.sum()),
            "stage_accuracy": float(accuracy_score(yL[teL], predL)),
            "stage_f1_macro": float(f1_score(yL[teL], predL, average="macro", zero_division=0)),
        }
    (OUT / "lookback_ablation.json").write_text(json.dumps(lookback_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 저수율 구간 (일별 say에서 주간 mean merge)
    bucket_block: dict = {"note": "홀드아웃 행의 4주 평균 저수율(lw4_mean_저수율)으로 삼분위 구간"}
    res_col = "lw4_mean_저수율(%)_mean"
    if res_col in meta_te.columns:
        q1, q2 = meta_te[res_col].quantile([0.33, 0.66])
        meta_te["저수율_bucket"] = pd.cut(
            meta_te[res_col],
            bins=[-np.inf, q1, q2, np.inf],
            labels=["low", "mid", "high"],
        )
        brows = []
        for lab in ["low", "mid", "high"]:
            m = meta_te["저수율_bucket"].astype(str) == lab
            ix = np.where(m.values)[0]
            if len(ix) < 2:
                continue
            brows.append(
                {
                    "bucket": lab,
                    "n": int(len(ix)),
                    "stage_accuracy": float(accuracy_score(y_st_te[ix], pred_st[ix])),
                    "binary_recall_0.5": float(
                        recall_score(y_bin_te[ix], (proba_bin[ix] >= 0.5).astype(int), pos_label=1, zero_division=0)
                    ),
                }
            )
        bucket_block["by_reservoir_level_tertile"] = brows
        pd.DataFrame(brows).to_csv(OUT / "metrics_by_reservoir_level_bucket.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input": str(wsp.SAY_CSV),
        "holdout_ratio": 0.15,
        "holdout_n": int(te.sum()),
        "model": "LightGBM (stage/reg/binary) — weekly_say와 유사 설정",
        "data_quality_daily": dq,
        "persistence_vs_model": persistence_block,
        "lookback_ablation": lookback_results,
        "reservoir_level": bucket_block,
        "outputs_dir": str(OUT),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3500])
    print(f"산출: {OUT}/")


if __name__ == "__main__":
    main()
