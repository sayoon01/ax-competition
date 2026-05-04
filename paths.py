"""레포 루트·입력 CSV·산출 디렉터리 단일 정의. 파이프라인·리포트·대시보드 스크립트가 공통으로 사용."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent
SAY_CSV = REPO / "finaldata_say.csv"

# --- 산출 (폴더명 변경 시 이 모듈만 수정) ---
OUT_WEEKLY_SAY = REPO / "outputs" / "weekly_say"
OUT_DAILY_D7 = REPO / "outputs" / "daily_d7"
OUT_WEEKLY_EXTRAS = REPO / "outputs" / "weekly_extras"
OUT_ANALYSIS = REPO / "outputs" / "analysis"
OUT_ASSIGNMENT_TOOLKIT = REPO / "outputs" / "assignment_toolkit"
OUT_REPORT = REPO / "outputs" / "report"

DASHBOARD_DIR = REPO / "dashboard"
