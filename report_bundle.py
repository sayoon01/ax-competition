#!/usr/bin/env python3
"""
주간·일 단위·extras 메트릭 JSON을 읽어 `outputs/report/report_metrics.md`를 만들고,
`scenario_recommendations_holdout.json`에서 시나리오별 활동 계획 버킷(키워드 추정) 매핑 표를 생성합니다.

실행: 파이프라인(주간·일 D+7, 선택 extras) 이후
  python3 report_bundle.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from paths import OUT_DAILY_D7, OUT_REPORT, OUT_WEEKLY_EXTRAS, OUT_WEEKLY_SAY, REPO

WEEKLY_METRICS = OUT_WEEKLY_SAY / "weekly_metrics.json"
DAILY_METRICS = OUT_DAILY_D7 / "pipeline_metrics.json"
EXTRAS_METRICS = OUT_WEEKLY_EXTRAS / "extras_metrics.json"
SCENARIO_JSON = OUT_WEEKLY_SAY / "scenario_recommendations_holdout.json"


def _load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_metrics_block(name: str, data: dict | None, missing_hint: str = "") -> str:
    if data is None:
        return f"### {name}\n\n*(파일 없음{missing_hint})*\n\n"
    lines = [f"### {name}\n"]
    if isinstance(data, dict):
        skip = {"metrics", "report"}
        for k, v in data.items():
            if k in skip:
                continue
            if isinstance(v, (dict, list)):
                lines.append(f"- **{k}**: `{json.dumps(v, ensure_ascii=False)[:500]}`")
            else:
                lines.append(f"- **{k}**: {v}")
        if "metrics" in data and isinstance(data["metrics"], dict):
            lines.append("\n**세부 metrics**:\n")
            for mk, mv in data["metrics"].items():
                if mk == "classification_stage" and isinstance(mv, dict) and "report" in mv:
                    mv = {k2: v2 for k2, v2 in mv.items() if k2 != "report"}
                if isinstance(mv, dict):
                    lines.append(f"- `{mk}`: " + ", ".join(f"{k2}={v2}" for k2, v2 in mv.items() if k2 != "report"))
                else:
                    lines.append(f"- `{mk}`: {mv}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_report_metrics_md() -> Path:
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    weekly = _load_json(WEEKLY_METRICS)
    daily = _load_json(DAILY_METRICS)
    extras = _load_json(EXTRAS_METRICS)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# 통합 지표 요약 (`report_metrics.md`)\n",
        f"*자동 생성: `report_bundle.py` — {ts}*\n",
        "원본: `outputs/weekly_say/weekly_metrics.json`, `outputs/daily_d7/pipeline_metrics.json`, `outputs/weekly_extras/extras_metrics.json`.\n",
        "---\n",
    ]
    parts.append(
        _fmt_metrics_block(
            "주간 본 파이프라인 (`weekly_metrics.json`)",
            weekly if isinstance(weekly, dict) else None,
            f": `{WEEKLY_METRICS.relative_to(REPO)}`",
        )
    )
    parts.append(
        _fmt_metrics_block(
            "일 단위 D+7 (`pipeline_metrics.json`)",
            daily if isinstance(daily, dict) else None,
            f": `{DAILY_METRICS.relative_to(REPO)}`",
        )
    )
    if extras is None:
        parts.append("### 보조 실험 (`extras_metrics.json`)\n\n*(파일 없음 — `pipeline_weekly_extras.py` 미실행 시 생략 가능)*\n\n")
    else:
        parts.append("### 보조 실험 (`extras_metrics.json`)\n\n")
        if isinstance(extras, dict):
            for k, v in extras.items():
                if k == "input_csv":
                    continue
                if isinstance(v, dict):
                    parts.append(f"- **`{k}`**: " + ", ".join(f"{k2}={v2}" for k2, v2 in v.items()) + "\n")
                else:
                    parts.append(f"- **`{k}`**: {v}\n")
        parts.append("\n")

    out = OUT_REPORT / "report_metrics.md"
    out.write_text("".join(parts), encoding="utf-8")
    return out


def _infer_activity_buckets(blob: str) -> str:
    """권고·제목·조건 텍스트에서 활동 계획 카테고리(키워드) 추정. 수동 보정은 CSV의 manual_* 열 사용."""
    t = blob
    tags: list[str] = []
    if any(k in t for k in ("유입", "강수", "호우", "방류", "수문", "혼합", "체류", "저수", "유량")):
        tags.append("물순환·방류·유입")
    if any(k in t for k in ("취수", "수심", "위치 조정", "취수장")):
        tags.append("취수·위치")
    if any(k in t for k in ("제거선", "인력 투입", "인력", "제거 ")):
        tags.append("제거선·인력")
    if any(k in t for k in ("활성탄", "약품", "응집", "PAC", "전처리", "응집제")):
        tags.append("약품·활성탄")
    if any(k in t for k in ("모니터링", "분석", "재예측", "발령", "종 ", "탁도", "클로로필", "pH")):
        tags.append("모니터링·분석")
    return "; ".join(sorted(set(tags))) if tags else "(키워드 미매칭 — 수동 매핑)"


def write_scenario_activity_maps() -> tuple[Path, Path]:
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    raw = _load_json(SCENARIO_JSON)
    rows: list[dict[str, str]] = []
    if not isinstance(raw, list):
        raw = []

    for item in raw:
        site = str(item.get("채수위치", ""))
        week = str(item.get("week_start_target", ""))
        scenarios = item.get("scenarios") or []
        if not isinstance(scenarios, list):
            continue
        for sc in scenarios:
            if not isinstance(sc, dict):
                continue
            sid = str(sc.get("id", ""))
            title = str(sc.get("title", ""))
            cond = str(sc.get("조건", ""))
            expected = str(sc.get("예상", ""))
            recs = sc.get("권고") or []
            rec_text = " | ".join(str(x) for x in recs) if isinstance(recs, list) else str(recs)
            blob = f"{title} {cond} {expected} {rec_text}"
            rows.append(
                {
                    "week_start_target": week,
                    "채수위치": site,
                    "scenario_id": sid,
                    "scenario_title": title,
                    "조건": cond,
                    "예상": expected,
                    "권고_전체": rec_text,
                    "inferred_activity_buckets": _infer_activity_buckets(blob),
                    "manual_activity_plan_메모": "",
                    "manual_담당_부서": "",
                }
            )

    csv_path = OUT_REPORT / "scenario_activity_map.csv"
    md_path = OUT_REPORT / "scenario_activity_map.md"
    fieldnames = list(rows[0].keys()) if rows else [
        "week_start_target",
        "채수위치",
        "scenario_id",
        "scenario_title",
        "조건",
        "예상",
        "권고_전체",
        "inferred_activity_buckets",
        "manual_activity_plan_메모",
        "manual_담당_부서",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    intro = (
        "# 시나리오 ↔ 활동 계획 매핑 표\n\n"
        "*`pipeline_weekly_say.py` 산출 `scenario_recommendations_holdout.json`을 펼친 표입니다.*\n\n"
        "- **`inferred_activity_buckets`**: 제목·조건·권고 문구 키워드로 자동 태깅(참고용).\n"
        "- **`manual_*` 열**: 보고서용으로 직접 채우기(정본 CSV는 `scenario_activity_map.csv`).\n\n"
        "---\n\n"
    )
    # MD: 앞 80행만 표로 (전체는 CSV)
    max_md_rows = 80
    slim = rows[:max_md_rows]
    if not slim:
        md_path.write_text(intro + "*(시나리오 JSON이 비었거나 없습니다.)*\n", encoding="utf-8")
        return csv_path, md_path

    cols = ["week_start_target", "채수위치", "scenario_id", "scenario_title", "inferred_activity_buckets", "manual_activity_plan_메모"]
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body_lines = []
    for r in slim:
        cells = []
        for c in cols:
            cell = str(r.get(c, "")).replace("|", "\\|").replace("\n", " ")
            if len(cell) > 60:
                cell = cell[:57] + "..."
            cells.append(cell)
        body_lines.append("| " + " | ".join(cells) + " |\n")
    note = f"\n\n*(표는 검증 구간 상위 {len(slim)}행만 표시. 전체는 `scenario_activity_map.csv` 참고.)*\n"
    md_path.write_text(intro + header + sep + "".join(body_lines) + note, encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    p1 = write_report_metrics_md()
    p2, p3 = write_scenario_activity_maps()
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")


if __name__ == "__main__":
    main()
