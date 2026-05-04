#!/usr/bin/env python3
"""outputs/*.json → dashboard/index.html 생성. 실행: python3 build_dashboard.py"""
from __future__ import annotations

import json
import re
import shutil
from html import escape
from pathlib import Path

from paths import DASHBOARD_DIR, OUT_DAILY_D7, OUT_WEEKLY_EXTRAS, OUT_WEEKLY_SAY

OUT_WEEKLY = OUT_WEEKLY_SAY
OUT_DAILY = OUT_DAILY_D7
OUT_EXTRAS = OUT_WEEKLY_EXTRAS
DASH_DIR = DASHBOARD_DIR
FIGURES_DIR = DASH_DIR / "_figures"

WEEKLY_FIG_SPECS: list[tuple[str, str]] = [
    ("shap_weekly_regression.png", "주간 회귀(log1p 세포수) SHAP"),
    ("shap_weekly_stage.png", "주간 발령단계 분류 SHAP"),
    ("shap_binary_subset_val_y1.png", "이진 모델 SHAP (검증·실제 관심↑)"),
    ("shap_binary_subset_val_y0.png", "이진 모델 SHAP (검증·실제 미발령)"),
    ("timeseries_prob_alert.png", "관심 이상 확률 시계열"),
]
DAILY_FIG_SPECS: list[tuple[str, str]] = [
    ("shap_mean_abs_bar.png", "일 단위 평균 |SHAP|"),
    ("shap_beeswarm_class.png", "일 단위 SHAP beeswarm"),
    ("timeseries_holdout_lightgbm.png", "D+7 발령단계 홀드아웃 시계열"),
]


def sync_dashboard_figures() -> None:
    """outputs/*.png → dashboard/_figures/… 복사. http.server --directory dashboard 로도 이미지 로드 가능."""
    if FIGURES_DIR.exists():
        shutil.rmtree(FIGURES_DIR)
    bundles: list[tuple[Path, str, list[tuple[str, str]]]] = [
        (OUT_WEEKLY, "weekly_say", WEEKLY_FIG_SPECS),
        (OUT_DAILY, "daily_d7", DAILY_FIG_SPECS),
    ]
    for src_root, sub, specs in bundles:
        dst_dir = FIGURES_DIR / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fn, _ in specs:
            src = src_root / fn
            if src.exists():
                shutil.copy2(src, dst_dir / fn)


def format_fig_panel(subdir: str, specs: list[tuple[str, str]]) -> str:
    """sync_dashboard_figures 이후 _figures/<subdir>/ 만 링크."""
    prefix = f"_figures/{subdir}"
    local = FIGURES_DIR / subdir
    pieces: list[str] = []
    for fn, caption in specs:
        if not (local / fn).exists():
            continue
        cap_e = escape(caption)
        src = f"{prefix}/{fn}"
        pieces.append(
            f'<figure class="fig-card"><img src="{escape(src)}" alt="{cap_e}" loading="lazy" />'
            f"<figcaption>{cap_e}</figcaption></figure>"
        )
    if not pieces:
        return (
            '<p class="meta">이 탭에 표시할 PNG가 없습니다. '
            "<code>pipeline_weekly_say.py</code> 또는 <code>pipeline_daily_d7.py</code> 실행 후 "
            "<code>python3 build_dashboard.py</code>로 다시 생성하세요.</p>"
        )
    return '<div class="fig-grid">' + "".join(pieces) + "</div>"


def load_json(p: Path, default):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def parse_hint_bars(summary: str) -> list[tuple[str, float]]:
    if not summary:
        return [("수온·수질·유량 종합", 100.0)]
    pairs: list[tuple[str, float]] = []
    for seg in summary.split(","):
        seg = seg.strip()
        m = re.search(r"(.+?)\s*(\d+)\s*회", seg)
        if m:
            pairs.append((m.group(1).strip(), float(m.group(2))))
    if not pairs:
        return [("모델 기여 요약", 100.0)]
    s = sum(w for _, w in pairs) or 1.0
    return [(n, round(100 * w / s, 1)) for n, w in pairs]


def site_snapshot(scenarios: list[dict], site: str) -> dict | None:
    rows = [r for r in scenarios if r.get("채수위치") == site]
    if not rows:
        return None
    rows.sort(key=lambda x: x.get("week_start_target", ""))
    last = rows[-1]
    p = float(last.get("p_alert_ge1_scenario") or last.get("p_alert_ge1_calibrated") or 0)
    stage = int(last.get("pred_stage", 0))
    stage_names = ["정상(미발령)", "관심", "경계", "대발생"]
    sn = stage_names[stage] if 0 <= stage < 4 else "—"
    if p < 0.3:
        band = "정상"
        band_cls = "band-ok"
    elif p < 0.5:
        band = "주의"
        band_cls = "band-warn"
    elif p < 0.7:
        band = "관심 가능성↑"
        band_cls = "band-alert"
    else:
        band = "경계 가능성↑"
        band_cls = "band-high"
    return {
        "site": site,
        "week": last.get("week_start_target"),
        "p": round(p * 100, 1),
        "stage": sn,
        "cyano": round(float(last.get("pred_cyano_max_approx") or 0), 0),
        "band": band,
        "band_cls": band_cls,
    }


def main() -> None:
    DASH_DIR.mkdir(parents=True, exist_ok=True)

    weekly = load_json(OUT_WEEKLY / "weekly_metrics.json", {})
    scenarios = load_json(OUT_WEEKLY / "scenario_recommendations_holdout.json", [])
    daily = load_json(OUT_DAILY / "pipeline_metrics.json", {})
    extras = load_json(OUT_EXTRAS / "extras_metrics.json", {})

    hint = weekly.get("metrics", {}).get("시차_특성_힌트_이진LGB", {})
    hint_summary = hint.get("요약", "") if isinstance(hint, dict) else ""
    hint_top = hint.get("상위5특성", "") if isinstance(hint, dict) else ""
    bars = parse_hint_bars(hint_summary)

    sites = ["문의", "추동", "회남"]
    cards = [site_snapshot(scenarios, s) for s in sites]
    cards = [c for c in cards if c]

    timeline = sorted(scenarios, key=lambda x: (x.get("week_start_target", ""), x.get("채수위치", "")))[-18:]

    sync_dashboard_figures()
    weekly_figs_html = format_fig_panel("weekly_say", WEEKLY_FIG_SPECS)
    daily_figs_html = format_fig_panel("daily_d7", DAILY_FIG_SPECS)

    payload = {
        "weekly_summary": {
            "rows": weekly.get("weekly_supervised_rows"),
            "features": weekly.get("feature_count"),
            "holdout": weekly.get("holdout_ratio"),
        },
        "model_metrics": weekly.get("metrics", {}),
        "daily_d7": daily,
        "extras": extras,
        "hint_top": hint_top,
        "bars": [{"label": a, "pct": b} for a, b in bars],
        "site_cards": cards,
        "timeline": timeline[-24:],
    }

    data_json = json.dumps(payload, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>대청댐 조류 위험 조기경보 대시보드</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #0c1222;
      --surface: #141c2f;
      --surface2: #1a2540;
      --border: rgba(94, 234, 212, 0.15);
      --text: #e8f1ff;
      --muted: #8ba3c7;
      --accent: #2dd4bf;
      --accent2: #38bdf8;
      --warn: #fbbf24;
      --danger: #f87171;
      --ok: #4ade80;
      --radius: 14px;
      --shadow: 0 18px 50px rgba(0,0,0,.45);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans KR", system-ui, sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, rgba(45,212,191,.12), transparent),
                  radial-gradient(800px 400px at 90% 0%, rgba(56,189,248,.1), transparent),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.55;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 60px; }}
    header {{
      display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
      gap: 16px; margin-bottom: 32px;
    }}
    h1 {{
      font-size: 1.65rem; font-weight: 700; margin: 0;
      letter-spacing: -0.02em;
      background: linear-gradient(120deg, var(--accent), var(--accent2));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }}
    .tagline {{ color: var(--muted); font-size: 0.95rem; max-width: 520px; }}
    .pill {{
      display: inline-flex; align-items: center; gap: 8px;
      padding: 8px 14px; border-radius: 999px;
      background: var(--surface2); border: 1px solid var(--border);
      font-size: 0.8rem; color: var(--muted);
    }}
    .pill strong {{ color: var(--accent); font-weight: 600; }}
    section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 22px 22px 26px;
      margin-bottom: 22px;
      box-shadow: var(--shadow);
    }}
    h2 {{
      font-size: 1.05rem; font-weight: 600; margin: 0 0 16px;
      color: var(--accent); display: flex; align-items: center; gap: 10px;
    }}
    h2 span.num {{
      display: inline-flex; width: 26px; height: 26px; border-radius: 8px;
      background: rgba(45,212,191,.15); color: var(--accent); font-size: 0.75rem;
      align-items: center; justify-content: center; font-weight: 700;
    }}
    .grid3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .card {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 18px 16px;
    }}
    .card h3 {{ margin: 0 0 12px; font-size: 1.1rem; font-weight: 600; }}
    .gauge {{
      height: 10px; border-radius: 6px; background: rgba(255,255,255,.06);
      overflow: hidden; margin: 10px 0 6px;
    }}
    .gauge > i {{
      display: block; height: 100%; border-radius: 6px;
      background: linear-gradient(90deg, var(--accent2), var(--accent));
      transition: width .6s ease;
    }}
    .meta {{ font-size: 0.82rem; color: var(--muted); }}
    .band {{
      display: inline-block; margin-top: 10px; padding: 4px 10px; border-radius: 8px;
      font-size: 0.78rem; font-weight: 600;
    }}
    .band-ok {{ background: rgba(74,222,128,.15); color: var(--ok); }}
    .band-warn {{ background: rgba(251,191,36,.12); color: var(--warn); }}
    .band-alert {{ background: rgba(248,113,113,.12); color: #fca5a5; }}
    .band-high {{ background: rgba(248,113,113,.22); color: var(--danger); }}
    .bar-row {{ margin-bottom: 12px; }}
    .bar-row label {{ display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px; color: var(--muted); }}
    .bar-row label b {{ color: var(--text); }}
    .bar-bg {{ height: 8px; border-radius: 5px; background: rgba(255,255,255,.06); overflow: hidden; }}
    .bar-fg {{ height: 100%; border-radius: 5px; background: linear-gradient(90deg, #0ea5e9, var(--accent)); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th, td {{ padding: 10px 8px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 500; font-size: 0.76rem; text-transform: uppercase; letter-spacing: .04em; }}
    .scenario-list {{ display: flex; flex-direction: column; gap: 12px; }}
    .sc-item {{
      border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
      background: rgba(0,0,0,.2);
    }}
    .sc-item h4 {{ margin: 0 0 6px; font-size: 0.92rem; color: var(--accent2); }}
    .sc-item p {{ margin: 0; font-size: 0.82rem; color: var(--muted); }}
    .sc-item ul {{ margin: 8px 0 0; padding-left: 18px; color: var(--text); font-size: 0.82rem; }}
    .cost-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
    .cost-item {{
      padding: 14px; border-radius: 10px; border: 1px dashed var(--border);
      font-size: 0.84rem; color: var(--muted);
    }}
    .cost-item strong {{ color: var(--accent); display: block; margin-bottom: 6px; font-size: 0.88rem; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
    .chip {{
      padding: 6px 12px; border-radius: 999px; background: rgba(56,189,248,.1);
      border: 1px solid rgba(56,189,248,.25); font-size: 0.78rem; color: var(--muted);
    }}
    html {{ scroll-behavior: smooth; }}
    section {{ scroll-margin-top: 72px; }}
    .dash-nav {{
      position: sticky; top: 0; z-index: 40;
      display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
      padding: 12px 0 16px; margin: -8px 0 8px;
      background: linear-gradient(180deg, rgba(12,18,34,.92) 70%, transparent);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(94,234,212,.08);
    }}
    .dash-nav a {{
      font-size: 0.78rem; font-weight: 500; color: var(--muted);
      text-decoration: none; padding: 8px 12px; border-radius: 999px;
      border: 1px solid transparent; transition: .15s ease;
    }}
    .dash-nav a:hover {{ color: var(--text); border-color: var(--border); background: var(--surface2); }}
    .meta-fold {{ margin: 0 0 14px; border-radius: 10px; border: 1px solid var(--border); background: rgba(0,0,0,.15); }}
    .meta-fold summary {{
      cursor: pointer; list-style: none; padding: 10px 14px; font-size: 0.82rem; color: var(--accent2); font-weight: 600;
    }}
    .meta-fold summary::-webkit-details-marker {{ display: none; }}
    .meta-fold summary::after {{ content: " ▼"; font-size: 0.65rem; opacity: .6; }}
    .meta-fold[open] summary::after {{ content: " ▲"; }}
    .meta-fold .meta {{ padding: 0 14px 12px; margin: 0; }}
    .stat-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 4px;
    }}
    .stat-tile {{
      background: var(--surface2); border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 14px 12px; border-left: 3px solid var(--accent2);
    }}
    .stat-tile .v {{ font-size: 1.35rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }}
    .stat-tile .k {{ font-size: 0.72rem; color: var(--muted); margin-top: 6px; line-height: 1.35; }}
    .site-card-inner {{ display: flex; gap: 16px; align-items: flex-start; }}
    .donut-wrap {{
      --p: 0; width: 64px; height: 64px; border-radius: 50%; flex-shrink: 0;
      background: conic-gradient(from -90deg, var(--accent2) calc(var(--p) * 1%), rgba(255,255,255,.07) 0);
      display: grid; place-items: center;
    }}
    .donut-wrap span {{
      width: 48px; height: 48px; border-radius: 50%; background: var(--surface2);
      display: grid; place-items: center; font-size: 0.7rem; font-weight: 700; color: var(--text);
      border: 1px solid var(--border);
    }}
    .tl-visual {{ margin-bottom: 18px; }}
    .tl-visual h3 {{ margin: 0 0 10px; font-size: 0.8rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
    .tl-item {{ display: flex; align-items: center; gap: 12px; margin: 8px 0; font-size: 0.78rem; color: var(--muted); }}
    .tl-track {{ flex: 1; min-width: 0; height: 10px; border-radius: 6px; background: rgba(255,255,255,.06); overflow: hidden; }}
    .tl-fill {{ height: 100%; border-radius: 6px; transition: width .5s ease; }}
    .tl-fill.low {{ background: linear-gradient(90deg, #22c55e, var(--accent)); }}
    .tl-fill.mid {{ background: linear-gradient(90deg, var(--accent), var(--warn)); }}
    .tl-fill.high {{ background: linear-gradient(90deg, var(--warn), var(--danger)); }}
    .tl-cap {{ flex: 0 0 120px; text-align: right; color: var(--text); font-variant-numeric: tabular-nums; }}
    .h2-ico {{ font-size: 1.1rem; opacity: .9; }}
    .fig-tabs {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
    .fig-tab {{
      padding: 10px 18px; border-radius: 10px; border: 1px solid var(--border);
      background: var(--surface2); color: var(--muted); cursor: pointer;
      font-family: inherit; font-size: 0.84rem; font-weight: 500; transition: .15s ease;
    }}
    .fig-tab:hover {{ color: var(--text); border-color: rgba(56,189,248,.35); }}
    .fig-tab.active {{ color: var(--text); border-color: var(--accent2); background: rgba(56,189,248,.12); }}
    .fig-panel {{ display: none; }}
    .fig-panel.is-active {{ display: block; }}
    .fig-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px;
    }}
    .fig-card {{
      margin: 0; border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
      background: rgba(0,0,0,.2);
    }}
    .fig-card img {{ width: 100%; height: auto; display: block; vertical-align: middle; }}
    .fig-card figcaption {{
      padding: 10px 12px; font-size: 0.78rem; color: var(--muted); border-top: 1px solid var(--border);
    }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 28px; }}
    @media (max-width: 640px) {{
      h1 {{ font-size: 1.35rem; }}
      .tl-cap {{ flex-basis: 88px; font-size: 0.72rem; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>대청댐 조류 위험 조기경보</h1>
        <p class="tagline">
          수집·기상·수문 데이터를 통합한 <strong>다음 주 관심 이상 확률</strong>과
          <strong>경보 단계 예측</strong>, 시나리오 기반 권고를 한 화면에서 확인합니다.
          (데이터: <code>finaldata_say.csv</code> 기반 주간 파이프라인 산출물)
        </p>
      </div>
      <div class="pill">
        <span>선행 예측</span>
        <strong>다음 주(주 단위)</strong>
        <span>· D+7 일모델은 별도</span>
      </div>
    </header>

    <nav class="dash-nav" aria-label="섹션 이동">
      <a href="#sec-sites">지점 위험도</a>
      <a href="#sec-lead">모델 지표</a>
      <a href="#sec-shap">영향 인자</a>
      <a href="#sec-figs">그래프</a>
      <a href="#sec-scenario">시나리오</a>
      <a href="#sec-cost">비용·리스크</a>
      <a href="#sec-timeline">타임라인</a>
    </nav>

    <section id="sec-sites">
      <h2><span class="num">1</span><span class="h2-ico" aria-hidden="true">◎</span> 지점별 위험도</h2>
      <details class="meta-fold">
        <summary>확률·등급이 어떻게 계산되나요?</summary>
        <p class="meta">확률은 검증 구간 기준 <strong>보정(isotonic)</strong> 또는 앙상블 값을 반영합니다. 등급: 정상 / 주의 / 관심·경계 가능성</p>
      </details>
      <div class="grid3" id="site-cards"></div>
    </section>

    <section id="sec-lead">
      <h2><span class="num">2</span><span class="h2-ico" aria-hidden="true">◇</span> 1~2주 선행 예측 개요</h2>
      <details class="meta-fold">
        <summary>주간 모델이 보는 데이터 범위</summary>
        <p class="meta">
          최근 <strong>4주 패널</strong> 특성으로 <strong>다음 주</strong> 발령 단계·세포 수 상한을 추정합니다.
          일 단위 모델은 <code>outputs/daily_d7/</code> 의 D+7 발령단계 예측을 참고하세요.
        </p>
      </details>
      <div class="stat-grid" id="metric-chips"></div>
    </section>

    <section id="sec-shap">
      <h2><span class="num">3</span><span class="h2-ico" aria-hidden="true">≋</span> 영향 인자 설명 (모델 기반 요약)</h2>
      <details class="meta-fold">
        <summary>막대 그래프가 의미하는 것</summary>
        <p class="meta">이진 LightGBM의 특성 중요도 상위를 <strong>시차 유형</strong>으로 묶은 비율입니다. (상세 SHAP: <code>outputs/weekly_say/</code> PNG)</p>
      </details>
      <div id="factor-bars"></div>
      <p class="meta" style="margin-top:14px" id="hint-top"></p>
    </section>

    <section id="sec-figs">
      <h2><span class="num">·</span><span class="h2-ico" aria-hidden="true">▣</span> SHAP · 시계열 (산출 PNG)</h2>
      <p class="meta" style="margin-top:-6px;margin-bottom:14px">
        빌드 시 <code>outputs/</code> PNG가 <code>dashboard/_figures/</code> 로 복사됩니다.
        미리보기: <code>python3 -m http.server 8765 --directory dashboard</code> 후 <code>http://127.0.0.1:8765/</code>
      </p>
      <div class="fig-tabs" role="tablist" aria-label="그래프 출처">
        <button type="button" class="fig-tab active" role="tab" aria-selected="true" data-tab="w">주간 (weekly_say)</button>
        <button type="button" class="fig-tab" role="tab" aria-selected="false" data-tab="d">일 D+7 (daily_d7)</button>
      </div>
      <div id="fig-panel-w" class="fig-panel is-active" role="tabpanel">{weekly_figs_html}</div>
      <div id="fig-panel-d" class="fig-panel" role="tabpanel">{daily_figs_html}</div>
    </section>

    <section id="sec-scenario">
      <h2><span class="num">4</span><span class="h2-ico" aria-hidden="true">▸</span> 시나리오 추천 (최근 검증 구간)</h2>
      <div class="scenario-list" id="scenario-box"></div>
    </section>

    <section id="sec-cost">
      <h2><span class="num">5</span><span class="h2-ico" aria-hidden="true">◆</span> 선제 대응 · 비용·리스크</h2>
      <div class="cost-grid">
        <div class="cost-item"><strong>긴급 채수·약품</strong>사전 예측으로 활성탄·응집제 과다 투입 완화 가능</div>
        <div class="cost-item"><strong>민원·신뢰</strong>조기 안내로 수돗물 불안 완화, 대응 인력 재배치 효율화</div>
        <div class="cost-item"><strong>운영 리스크</strong>방류·취수 심도 조정 검토 시점을 앞당겨 결정 지원</div>
      </div>
    </section>

    <section id="sec-timeline">
      <h2 style="color:var(--accent2)"><span class="h2-ico" aria-hidden="true">⏱</span> 다음 주 타임라인 (검증 구간 샘플)</h2>
      <p class="meta" style="margin-top:-6px;margin-bottom:12px">막대 길이 = 관심 이상 확률(%). 표와 동일 데이터입니다.</p>
      <div class="tl-visual" id="timeline-visual" aria-hidden="true"></div>
      <div style="overflow-x:auto">
        <table><thead><tr><th>주 시작</th><th>지점</th><th>관심↑확률(%)</th><th>예측단계</th><th>예측세포수</th></tr></thead><tbody id="timeline-body"></tbody></table>
      </div>
    </section>

    <footer>
      정적 스냅샷 · <code>python3 build_dashboard.py</code> 로 재생성 ·
      미리보기: <code>python3 -m http.server 8765 --directory dashboard</code> → <code>http://127.0.0.1:8765/</code>
      (그래프는 <code>_figures/</code> 복사본)
    </footer>
  </div>

  <script type="application/json" id="dash-payload">{data_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('dash-payload').textContent);

    function esc(s) {{
      return String(s == null ? '' : s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}

    // 1) 지점 카드
    const siteRoot = document.getElementById('site-cards');
    (DATA.site_cards || []).forEach(c => {{
      const div = document.createElement('div');
      div.className = 'card';
      const pw = Math.min(100, Number(c.p) || 0);
      div.innerHTML = `
        <h3>${{esc(c.site)}}</h3>
        <div class="site-card-inner">
          <div class="donut-wrap" style="--p:${{pw}}"><span>${{c.p}}%</span></div>
          <div style="flex:1;min-width:0">
            <div class="meta">기준 주 ${{esc(c.week)}}</div>
            <div class="gauge"><i style="width:${{pw}}%"></i></div>
            <div class="meta">관심 이상 · 예측 <strong style="color:var(--text)">${{esc(c.stage)}}</strong></div>
            <div class="meta">세포수(대략) <strong>${{esc(c.cyano)}}</strong> cells/mL</div>
            <span class="band ${{esc(c.band_cls)}}">${{esc(c.band)}}</span>
          </div>
        </div>`;
      siteRoot.appendChild(div);
    }});
    if (!DATA.site_cards || !DATA.site_cards.length) {{
      siteRoot.innerHTML = '<p class="meta">지점 카드 데이터 없음 — pipeline_weekly_say.py 실행 후 build_dashboard.py 를 다시 실행하세요.</p>';
    }}

    // 2) 지표 타일
    const chips = document.getElementById('metric-chips');
    const m = DATA.model_metrics || {{}};
    const addTile = (label, val) => {{
      const s = document.createElement('div');
      s.className = 'stat-tile';
      s.innerHTML = '<div class="v">' + esc(val) + '</div><div class="k">' + esc(label) + '</div>';
      chips.appendChild(s);
    }};
    if (m.binary_calibrated_isotonic_lgb) {{
      addTile('보정 이진 ROC-AUC', (m.binary_calibrated_isotonic_lgb.roc_auc || 0).toFixed(3));
      addTile('관심↑ 재현율', (m.binary_calibrated_isotonic_lgb.recall_alert || 0).toFixed(2));
    }}
    if (m.classification_stage) {{
      addTile('단계 정확도', (m.classification_stage.accuracy || 0).toFixed(2));
    }}
    if (m.regression_log1p_cyano) {{
      addTile('세포수 회귀 R²', (m.regression_log1p_cyano.r2 || 0).toFixed(2));
    }}
    const dd = DATA.daily_d7 || {{}};
    if (dd.metrics && dd.metrics.lightgbm) {{
      addTile('일단위 D+7 정확도', (dd.metrics.lightgbm.accuracy || 0).toFixed(2));
    }}

    // 3) 바
    const fb = document.getElementById('factor-bars');
    (DATA.bars || []).forEach(b => {{
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML = `<label><span>${{esc(b.label)}}</span><b>${{b.pct}}%</b></label>
        <div class="bar-bg"><div class="bar-fg" style="width:${{b.pct}}%"></div></div>`;
      fb.appendChild(row);
    }});
    document.getElementById('hint-top').textContent = DATA.hint_top
      ? '상위 특성: ' + DATA.hint_top
      : '';

    // 4) 시나리오: 최근 6행만 카드
    const scBox = document.getElementById('scenario-box');
    const tl = (DATA.timeline || []).slice(-6);
    tl.forEach(row => {{
      const scs = row.scenarios || [];
      if (!scs.length) return;
      scs.slice(0, 2).forEach(sc => {{
        const el = document.createElement('div');
        el.className = 'sc-item';
        const ul = (sc.권고 || []).map(x => '<li>' + esc(x) + '</li>').join('');
        el.innerHTML = `<h4>${{esc(row.week_start_target)}} · ${{esc(row.채수위치)}} — ${{esc(sc.title)}}</h4>
          <p><strong>조건</strong> ${{esc(sc.조건)}}</p>
          <p><strong>예상</strong> ${{esc(sc.예상)}}</p>
          <ul>${{ul}}</ul>`;
        scBox.appendChild(el);
      }});
    }});

    // 타임라인 막대 + 테이블
    const tlRows = (DATA.timeline || []).slice(-14);
    const tv = document.getElementById('timeline-visual');
    const tvTitle = document.createElement('h3');
    tvTitle.textContent = '관심 이상 확률 막대 (최근 ' + tlRows.length + '건)';
    tv.appendChild(tvTitle);
    tlRows.forEach(row => {{
      const p = (row.p_alert_ge1_scenario != null ? row.p_alert_ge1_scenario : row.p_alert_ge1_calibrated) || 0;
      const pct = Math.min(100, p * 100);
      let fcls = 'low';
      if (pct >= 50) fcls = 'high';
      else if (pct >= 30) fcls = 'mid';
      const div = document.createElement('div');
      div.className = 'tl-item';
      div.innerHTML = '<div class="tl-track"><div class="tl-fill ' + fcls + '" style="width:' + pct.toFixed(1) + '%"></div></div>'
        + '<div class="tl-cap">' + esc(row.week_start_target) + ' · ' + esc(row.채수위치)
        + ' <strong>' + pct.toFixed(1) + '%</strong></div>';
      tv.appendChild(div);
    }});

    const tb = document.getElementById('timeline-body');
    tlRows.forEach(row => {{
      const tr = document.createElement('tr');
      const p = (row.p_alert_ge1_scenario != null ? row.p_alert_ge1_scenario : row.p_alert_ge1_calibrated) || 0;
      tr.innerHTML = `<td>${{esc(row.week_start_target)}}</td><td>${{esc(row.채수위치)}}</td>
        <td>${{(p*100).toFixed(1)}}</td><td>${{esc(row.pred_stage)}}</td>
        <td>${{Math.round(row.pred_cyano_max_approx || 0)}}</td>`;
      tb.appendChild(tr);
    }});

    (function() {{
      const tabs = document.querySelectorAll('.fig-tab');
      const panels = document.querySelectorAll('.fig-panel');
      tabs.forEach(btn => {{
        btn.addEventListener('click', () => {{
          tabs.forEach(b => {{ b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); }});
          panels.forEach(p => {{ p.classList.remove('is-active'); }});
          btn.classList.add('active');
          btn.setAttribute('aria-selected', 'true');
          const id = btn.getAttribute('data-tab') === 'w' ? 'fig-panel-w' : 'fig-panel-d';
          const el = document.getElementById(id);
          if (el) el.classList.add('is-active');
        }});
      }});
    }})();
  </script>
</body>
</html>"""

    out_path = DASH_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"작성: {out_path}")


if __name__ == "__main__":
    main()
