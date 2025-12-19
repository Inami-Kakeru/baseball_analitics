from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_japanese_matplotlib() -> None:
    """
    Windows想定で日本語ゴシックを優先。
    無い場合はDejaVu等にフォールバック（文字化けの可能性はあるのでログで分かるようにする）。
    """
    preferred = ["Yu Gothic", "Meiryo", "MS Gothic", "Yu Gothic UI", "Noto Sans CJK JP", "IPAexGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), None)
    # 見つかったものだけを指定（findfont警告のスパムを防ぐ）
    mpl.rcParams["font.family"] = [chosen] if chosen else preferred
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 100


def save_png_300dpi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")


def add_footer(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.0,
        -0.18,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#333333",
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


