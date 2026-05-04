#!/usr/bin/env python3
"""
finaldata_say.csv 기반 추가 분석 (기존 weekly_say 파이프라인 삭제 없음)

- RandomForest: 발령단계 분류 + log1p(다음주 세포수) 회귀 (동일 시간 홀드아웃)
- XGBoost + GridSearchCV(TimeSeriesSplit): 단계 분류 소규모 튜닝
- LSTM(Keras): 주간 집계 시계열에서 (4주 × 변수) → 다음 주 total_cyano_max (회귀), 시간 순 분할

산출: outputs/weekly_extras/extras_metrics.json (+ 선택 lstm_history.csv)
실행: python3 pipeline_weekly_extras.py
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import pipeline_weekly_say as wsp
from paths import OUT_WEEKLY_EXTRAS as OUT_DIR

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def lstm_sequences_from_weekly(w: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """지점별 주간 표 → (n, 4, F) 시퀀스, log1p(다음 주 total_cyano_max), 해당 타깃 주 시작일."""
    drop = {"채수위치", "week", "week_start"}
    feat_cols = [c for c in w.columns if c not in drop and pd.api.types.is_numeric_dtype(w[c])]
    if "total_cyano_max" not in w.columns:
        raise ValueError("weekly 테이블에 total_cyano_max 필요")

    Xs: list[np.ndarray] = []
    ys: list[float] = []
    dates: list[np.datetime64] = []

    for site in w["채수위치"].unique():
        sub_full = w[w["채수위치"] == site].sort_values("week_start").reset_index(drop=True)
        tmax2 = sub_full["total_cyano_max"].astype(float).values
        arr2 = (
            sub_full[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
            .fillna(sub_full[feat_cols].median(numeric_only=True))
            .fillna(0.0)
            .values.astype(np.float32)
        )
        for i in range(len(sub_full) - 4):
            Xs.append(arr2[i : i + 4])
            ys.append(np.log1p(max(float(tmax2[i + 4]), 0.0)))
            dates.append(np.datetime64(pd.Timestamp(sub_full["week_start"].iloc[i + 4])))

    Xa = np.stack(Xs, axis=0)
    Xa = np.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)
    return Xa, np.array(ys, dtype=np.float32), np.array(dates)


def run_lstm(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    holdout: float = 0.15,
    epochs: int = 60,
    batch_size: int = 32,
) -> dict:
    import tensorflow as tf

    tf.random.set_seed(42)
    order = np.argsort(dates)
    Xo = X[order]
    yo = y[order]
    cut = int(len(Xo) * (1.0 - holdout))
    cut = max(cut, 50)
    X_tr, X_te = Xo[:cut], Xo[cut:]
    y_tr, y_te = yo[:cut], yo[cut:]
    y_tr = np.nan_to_num(y_tr, nan=0.0)
    y_te = np.nan_to_num(y_te, nan=0.0)

    F = X_tr.shape[2]
    scaler = MinMaxScaler()
    flat_tr = np.nan_to_num(X_tr.reshape(-1, F), nan=0.0, posinf=0.0, neginf=0.0)
    scaler.fit(flat_tr)
    X_tr_s = scaler.transform(flat_tr).reshape(X_tr.shape[0], 4, F)
    flat_te = np.nan_to_num(X_te.reshape(-1, F), nan=0.0, posinf=0.0, neginf=0.0)
    X_te_s = scaler.transform(flat_te).reshape(X_te.shape[0], 4, F)
    X_tr_s = np.nan_to_num(X_tr_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_s = np.nan_to_num(X_te_s, nan=0.0, posinf=0.0, neginf=0.0)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4, F)),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    hist = model.fit(
        X_tr_s,
        y_tr,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )
    y_pred = model.predict(X_te_s, verbose=0).flatten()
    mae_log = float(mean_absolute_error(y_te, y_pred))
    rmse_log = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    r2 = float(r2_score(y_te, y_pred))
    mae_cells = float(mean_absolute_error(np.expm1(y_te), np.maximum(np.expm1(y_pred), 0.0)))
    return {
        "lstm_mae_log": mae_log,
        "lstm_rmse_log": rmse_log,
        "lstm_r2_log": r2,
        "lstm_mae_cells_approx": mae_cells,
        "lstm_epochs_ran": len(hist.history["loss"]),
        "lstm_n_train": int(len(X_tr_s)),
        "lstm_n_test": int(len(X_te_s)),
        "lstm_features": int(F),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = wsp._read_say()
    tbl = wsp.build_weekly_supervised(df)
    tbl = tbl.dropna(subset=["y_cyano_max_next", "y_stage_max_next"]).reset_index(drop=True)
    feats = wsp.feature_cols(tbl)
    tbl[feats] = tbl[feats].replace([np.inf, -np.inf], np.nan).fillna(tbl[feats].median(numeric_only=True))
    le = pd.factorize(tbl["채수위치"])[0]
    tbl["site_code"] = le
    feat_all = feats + ["site_code"]
    X = tbl[feat_all].values
    y_stage = tbl["y_stage_max_next"].clip(0, 3).astype(int).values
    y_reg = np.log1p(np.maximum(tbl["y_cyano_max_next"].values, 0.0))
    dates = pd.to_datetime(tbl["week_start_target"])
    tr, te = wsp.time_mask(dates, 0.15)
    X_tr, X_te = X[tr], X[te]
    y_st_tr, y_st_te = y_stage[tr], y_stage[te]
    y_reg_tr, y_reg_te = y_reg[tr], y_reg[te]

    metrics: dict = {"input_csv": str(wsp.SAY_CSV)}

    # --- RandomForest
    rfc = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rfc.fit(X_tr, y_st_tr)
    prf = rfc.predict(X_te)
    metrics["random_forest_classifier_stage"] = {
        "accuracy": float(accuracy_score(y_st_te, prf)),
        "f1_macro": float(f1_score(y_st_te, prf, average="macro", zero_division=0)),
    }

    rfr = RandomForestRegressor(n_estimators=250, max_depth=14, random_state=42, n_jobs=-1)
    rfr.fit(X_tr, y_reg_tr)
    prr = rfr.predict(X_te)
    metrics["random_forest_regressor_log1p_cyano"] = {
        "r2": float(r2_score(y_reg_te, prr)),
        "mae_log": float(mean_absolute_error(y_reg_te, prr)),
        "mae_cells_approx": float(
            mean_absolute_error(np.expm1(y_reg_te), np.maximum(np.expm1(prr), 0.0))
        ),
    }

    # --- XGBoost GridSearch (시계열 CV, 학습 구간만)
    try:
        from xgboost import XGBClassifier

        order = np.argsort(dates.values[tr])
        X_ts = X_tr[order]
        y_ts = y_st_tr[order]
        nc = int(np.max(y_ts)) + 1
        nc = max(nc, 3)
        param_grid = {
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [200, 350],
            "subsample": [0.85],
            "colsample_bytree": [0.85],
        }
        xgb_base = XGBClassifier(
            objective="multi:softprob",
            num_class=nc,
            random_state=42,
            verbosity=0,
        )
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(
            xgb_base,
            param_grid,
            cv=tscv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_ts, y_ts)
        best = grid.best_estimator_
        pred_g = best.predict(X_te)
        metrics["xgboost_gridsearch_stage"] = {
            "best_params": grid.best_params_,
            "best_cv_f1_macro": float(grid.best_score_),
            "holdout_accuracy": float(accuracy_score(y_st_te, pred_g)),
            "holdout_f1_macro": float(f1_score(y_st_te, pred_g, average="macro", zero_division=0)),
        }
    except Exception as e:
        metrics["xgboost_gridsearch_error"] = str(e)

    # --- LSTM: 주간 원시 집계 → 4주 창
    lstm_metrics: dict = {"skipped": True, "reason": ""}
    try:
        w = wsp.daily_to_weekly(df)
        Xseq, yseq, dseq = lstm_sequences_from_weekly(w)
        if len(Xseq) < 120:
            lstm_metrics = {"skipped": True, "reason": "시퀀스 샘플 부족"}
        else:
            lstm_metrics = run_lstm(Xseq, yseq, dseq)
    except Exception as e:
        lstm_metrics = {"skipped": True, "reason": str(e)}

    metrics["lstm_weekly_sequence"] = lstm_metrics

    (OUT_DIR / "extras_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"저장: {OUT_DIR / 'extras_metrics.json'}")


if __name__ == "__main__":
    main()
