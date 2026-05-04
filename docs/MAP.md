# 프로젝트 한 장 맵

`finaldata_say.csv` 하나를 입력으로 두고, **스크립트 이름 = 역할**이 드러나도록 정리했습니다. 산출 **폴더명**(`outputs/weekly_say` 등)은 기존과 동일합니다.

---

## 1. 실행 순서 (복붙용)

```text
python3 pipeline_weekly_say.py
python3 pipeline_daily_d7.py
python3 pipeline_weekly_extras.py          # 선택
python3 pipeline_analysis.py               # 선택
python3 pipeline_assignment_toolkit.py     # 선택
python3 report_bundle.py
python3 build_dashboard.py
```

---

## 2. 스크립트 ↔ 산출 (이름만 외우면 됨)

| 순서 | 스크립트 | 산출 디렉터리 |
|------|-----------|----------------|
| ① | `pipeline_weekly_say.py` | `outputs/weekly_say/` |
| ② | `pipeline_daily_d7.py` | `outputs/daily_d7/` |
| ③ | `pipeline_weekly_extras.py` | `outputs/weekly_extras/` |
| ④ | `pipeline_analysis.py` | `outputs/analysis/` |
| ⑤ | `pipeline_assignment_toolkit.py` | `outputs/assignment_toolkit/` |
| ⑥ | `report_bundle.py` | `outputs/report/` |
| ⑦ | `build_dashboard.py` | `dashboard/` + `dashboard/_figures/` |

**공통**: `paths.py` (모든 `outputs/...` 경로 단일 정의), `plot_config.py` (matplotlib 한글).

---

## 3. 구 이름 → 새 이름 (검색·히스토리용)

| 이전 | 현재 |
|------|------|
| `weekly_say_pipeline.py` | `pipeline_weekly_say.py` |
| `run_algae_pipeline.py` | `pipeline_daily_d7.py` |
| `weekly_extras_rf_xgbgrid_lstm.py` | `pipeline_weekly_extras.py` |
| `analysis_diagnostics.py` | `pipeline_analysis.py` |
| `run_assignment_toolkit.py` | `pipeline_assignment_toolkit.py` |
| `build_submission_report.py` | `report_bundle.py` |

---

## 4. 디렉터리 트리 (요약)

```text
ax-competition/
  finaldata_say.csv          # 유일 입력 (학습·분석 스크립트)
  paths.py                   # REPO, SAY_CSV, outputs/* 경로
  pipeline_*.py              # ①~⑤
  report_bundle.py           # ⑥
  build_dashboard.py         # ⑦
  plot_config.py
  outputs/
    weekly_say/              # 주간 본 파이프라인
    daily_d7/                # 일 D+7
    weekly_extras/           # RF·Grid·LSTM 보조
    analysis/                # 진단
    assignment_toolkit/      # 과제 보강
    report/                  # 통합 md + 시나리오 매핑
  dashboard/
  docs/
    MAP.md                   # 이 파일
    STRUCTURE.md             # 겹침·역할 상세
    pipelines/*.md           # 스크립트별 설명
```

---

## 5. 상세 문서

- 겹침·왜 폴더가 나뉘는지: [`STRUCTURE.md`](STRUCTURE.md)
- 스크립트별 흐름: [`pipelines/pipeline_weekly_say.md`](pipelines/pipeline_weekly_say.md) 등 동일 접두 파일명
- 산출 파일 목록: [`../outputs/README.txt`](../outputs/README.txt)
