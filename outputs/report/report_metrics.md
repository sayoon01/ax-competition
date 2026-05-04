# 통합 지표 요약 (`report_metrics.md`)
*자동 생성: `report_bundle.py` — 2026-05-04 06:57 UTC*
원본: `outputs/weekly_say/weekly_metrics.json`, `outputs/daily_d7/pipeline_metrics.json`, `outputs/weekly_extras/extras_metrics.json`.
---
### 주간 본 파이프라인 (`weekly_metrics.json`)

- **source**: /home/keti_spark1/yune/ax-competition/finaldata_say.csv
- **입력_데이터**: finaldata_say.csv 만 사용 (원본 finaldata.csv 는 이 스크립트에서 읽지 않음)
- **weekly_supervised_rows**: 1512
- **holdout_ratio**: 0.15
- **feature_count**: 76

**세부 metrics**:

- `regression_log1p_cyano`: r2=0.9568324079507321, mae_log=0.6039855224091847, mae_cells_approx=4172.678838388159
- `classification_stage`: accuracy=0.7236842105263158, f1_macro=0.5707870934633396
- `classification_stage_ensemble_lgb_xgb`: accuracy=0.7236842105263158, f1_macro=0.5707870934633396
- `binary_exogenous_only_lgb`: n_features=49, roc_auc=0.9754962779156328, recall_alert=0.75
- `binary_lgb`: roc_auc=0.9959677419354839, accuracy=0.956140350877193, recall_alert=0.9134615384615384, precision_alert=0.9895833333333334
- `binary_xgb`: roc_auc=0.9962003722084367
- `binary_ensemble_avg_proba_lgb_xgb`: roc_auc=0.9958901985111662, accuracy=0.9649122807017544, f1=0.9603960396039604, recall_alert=0.9326923076923077, precision_alert=0.9897959183673469
- `binary_calibrated_isotonic_lgb`: roc_auc=0.9948045905707196, accuracy=0.9649122807017544, f1=0.9607843137254902, recall_alert=0.9423076923076923, precision_alert=0.98
- `operational_boundary_f1`: 0.22857142857142856
- `시차_특성_힌트_이진LGB`: 요약=4주평균 4회, 직전1주(lw1) 3회, 4주추세(slope) 2회, 기타 1회, 상위5특성=lw1_total_cyano_mean, lw1_total_cyano_max, lw4_mean_일강수_주합_mm, lw4_std_DO(㎎/L)_mean, lw1_수온(℃)_mean

### 일 단위 D+7 (`pipeline_metrics.json`)

- **input_csv**: /home/keti_spark1/yune/ax-competition/finaldata_say.csv
- **input_rows**: 10920
- **modeling_rows**: 10899
- **target**: target_발령단계_D7_코드
- **holdout_ratio**: 0.15
- **features_n**: 58
- **classes**: `["미발령", "관심", "경계"]`
- **발령단계_vs_세포수_일치율**: 1.0

**세부 metrics**:

- `lightgbm`: accuracy=0.7529051987767584, f1_macro=0.5518900558540262
- `xgboost`: accuracy=0.7486238532110092, f1_macro=0.5445098313244081

### 보조 실험 (`extras_metrics.json`)

- **`random_forest_classifier_stage`**: accuracy=0.6929824561403509, f1_macro=0.5173163538368762
- **`random_forest_regressor_log1p_cyano`**: r2=0.9629170437876495, mae_log=0.5631039681510696, mae_cells_approx=4799.314473189832
- **`xgboost_gridsearch_stage`**: best_params={'colsample_bytree': 0.85, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 200, 'subsample': 0.85}, best_cv_f1_macro=0.7673623436510938, holdout_accuracy=0.6973684210526315, holdout_f1_macro=0.5304646057040038
- **`lstm_weekly_sequence`**: lstm_mae_log=1.2682405710220337, lstm_rmse_log=1.6365618573650609, lstm_r2_log=0.8205393552780151, lstm_mae_cells_approx=8272.2021484375, lstm_epochs_ran=17, lstm_n_train=1318, lstm_n_test=233, lstm_features=19

