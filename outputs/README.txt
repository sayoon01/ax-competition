  ▶ 짧은 맵: docs/MAP.md
  ▶ 구조·겹침 상세: docs/STRUCTURE.md
  ▶ pyenv 3.14 _ctypes: docs/PYTHON_ENV.md → 권장: /usr/bin/python3.10 …

[입력 데이터]
  finaldata_say.csv 는 노트북 등 전처리(예: data_preprocessing.ipynb)에서 생성합니다.
  pipeline_daily_d7.py / pipeline_weekly_say.py / pipeline_weekly_extras.py 는 say 를 덮어쓰지 않습니다.

[dashboard/]
  build_dashboard.py 실행 시 index.html 생성
  - outputs/weekly_say/*.json, daily_d7/pipeline_metrics.json, weekly_extras/extras_metrics.json 을 읽어
    지점별 위험도·선행예측 요약·영향인자 막대·시나리오·비용 블록을 렌더링 (데이터는 HTML에 인라인)
  - 미리보기: python3 -m http.server 8765 --directory dashboard 후 http://127.0.0.1:8765/
    (build_dashboard.py 가 SHAP 등 PNG를 dashboard/_figures/ 로 복사)

outputs 폴더 구조 (파일 역할이 겹치지 않도록 하위 폴더로 분리)

[outputs/daily_d7/]
  pipeline_daily_d7.py 가 생성합니다.
  - finaldata_say.csv 기반 일 단위 특성 → D+7일 발령단계 예측(LightGBM/XGBoost)
  - pipeline_metrics.json, holdout_predictions_daily_d7.csv
  - shap_*.png, timeseries_holdout_lightgbm.png

[outputs/weekly_say/]
  pipeline_weekly_say.py 가 생성합니다. (주 단위)
  - weekly_supervised_rows.csv, weekly_metrics.json, scenario_recommendations_holdout.json
  - holdout_predictions_weekly.csv
  - shap_*.png, timeseries_prob_alert.png

[outputs/report/]
  report_bundle.py 가 생성합니다. (metrics JSON + scenario JSON 읽기만)
  - report_metrics.md, scenario_activity_map.csv / .md

[outputs/weekly_extras/]
  pipeline_weekly_extras.py 가 생성합니다.
  - extras_metrics.json (+ 선택 lstm_history.csv)

[outputs/assignment_toolkit/]
  pipeline_assignment_toolkit.py 가 생성합니다. (과제 보고 보강용)
  - run_log.json, profile/, baselines/, quantile/, pdp/, cost/, optimization/, plotly/, shap_interaction/

[outputs/analysis/]
  pipeline_analysis.py 가 생성합니다. (홀드아웃 15% = 주간 파이프라인과 동일 분할)
  - analysis_summary.json, confusion_matrix_*, metrics_by_*, threshold_*, lookback_ablation.json 등

한글 그래프: plot_config.py 의 setup_korean_matplotlib() 를 파이프라인이 공통 사용합니다.
