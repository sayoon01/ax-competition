# 과제 보강 툴킷 (`pipeline_assignment_toolkit.py`)

과제 개요의 **① 예측·② 영향인자·의사결정**을 보강하기 위한 **별도 산출물**입니다. 핵심 파이프라인(`weekly_say` 등)을 **대체하지 않으며**, `outputs/assignment_toolkit/`에만 씁니다.

## 실행

```bash
pip install -r requirements.txt
python3 pipeline_assignment_toolkit.py
```

**Python 3.14:** `ydata-profiling`은 PyPI에 3.14용 휠이 없어 `requirements.txt`에서 `python_version < "3.14"` 조건으로 빠집니다. ①단계는 자동으로 **간이 HTML**(describe·결측)만 생성합니다. 전체 리포트가 필요하면 **3.10~3.13** 가상환경에서 `pip install "ydata-profiling>=4.6"` 후 같은 스크립트를 실행하세요.

**`ModuleNotFoundError: No module named '_ctypes'`** 가 나면 pyenv 빌드가 깨진 것입니다. **`docs/PYTHON_ENV.md`** 를 보고 시스템 Python 3.10 사용 또는 `libffi-dev` 후 pyenv 재설치를 하세요.

한 단계가 실패해도 다음 단계는 `_safe`로 계속 진행합니다. 요약은 `outputs/assignment_toolkit/run_log.json` 입니다.

## 단계별 산출

| 순서 | 내용 | 주요 파일 |
|------|------|-----------|
| ① | `ydata-profiling` 프로파일 HTML (실패 시 describe 기반 간이 HTML) | `profile/say_profile.html` |
| ② | `statsmodels` 지수평활 vs 주간 naive persistence (지점별 MAE) | `baselines/weekly_cyano_statsmodels.*` |
| ③ | LightGBM **분위수 회귀** 0.1/0.5/0.9, 구간 커버리지·pinball | `quantile/quantile_lgb_summary.json`, `quantile_holdout_predictions.csv` |
| ④ | **PDP** 2특성 (`sklearn.inspection.PartialDependenceDisplay`) | `pdp/pdp_top2_lgb_regression.png` |
| ⑤ | 이진 LGB **FP/FN 가중 비용** + 임계값 곡선 | `cost/threshold_cost_operational.csv`, `threshold_cost_curve.png` |
| ⑥ | **PuLP** 이진 예시 (모니터링 vs 조치 최소비용) | `optimization/pulp_example.json` |
| ⑦ | **Plotly** 검증 구간 확률·예측 세포 HTML | `plotly/holdout_prob_alert.html`, `holdout_pred_cyano.html` |
| ⑧ | **SHAP 상호작용** 상위 특성 쌍 (회귀 LGB 소표본) | `shap_interaction/shap_interaction_top_pairs.json` |

## 제안 대비 매핑

- **시계열 베이스라인**: `sktime` 대신 **statsmodels `ExponentialSmoothing`** (의존성 단순화).
- **구간 예측**: **NGBoost** 대신 **LightGBM `objective=quantile`** (동일 목적).
- **재현·MLOps**: 본 레포는 **`requirements.txt` + 고정 시드 + 실행 순서**로 정리. DVC는 미도입.

## `pipeline_analysis.py` 와 차이

| 항목 | `pipeline_analysis` | `pipeline_assignment_toolkit` |
|------|------------------------|---------------------------|
| 초점 | 혼동행렬·지점/계절·룩백·일별 품질 | 프로파일·전통 시계열·분위수·PDP·비용 LP·Plotly·SHAP interaction |
| 산출 | `outputs/analysis/` | `outputs/assignment_toolkit/` |

둘 다 주간 홀드아웃 개념을 쓰지만 **폴더가 분리**되어 있어 제출물에서 역할을 나누기 쉽습니다.
