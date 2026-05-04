"""matplotlib 한글 폰트 — pipeline_daily_d7 / pipeline_weekly_say 등 공통."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def setup_korean_matplotlib() -> None:
    """나눔고딕 등 등록 후 rcParams 적용. SHAP/서브플롯 전에 호출."""
    candidates = [
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"),
    ]
    for font_path in candidates:
        if not font_path.exists():
            continue
        try:
            fm.fontManager.addfont(str(font_path))
            name = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
