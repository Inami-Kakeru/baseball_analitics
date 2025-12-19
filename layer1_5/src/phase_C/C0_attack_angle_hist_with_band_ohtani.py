from __future__ import annotations

"""
実験1（最優先）:
attack_angleの分布＋固定帯域の可視化（“帯域って何？”を一発で理解させる）

出力:
- layer1_5/data/output/plots/phase_C/C0_attack_angle_hist_with_band_ohtani.png

仕様:
- 大谷（overall / two_strike）を上下2段
- 同じ図に固定帯域 [lower, upper] を半透明帯で重ねる
- 図右上に n_total / n_attack_angle_available / 欠損率 を表示
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONTACT_PATH = Path("data/intermediate/contact_quality.csv")
BAND_PATH = Path("data/output/ohtani_attack_angle_band_rate.csv")
OUT_PNG = Path("data/output/plots/phase_C/C0_attack_angle_hist_with_band_ohtani.png")


def _must_have(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} 必須列が不足: {missing}")


def _load_band() -> tuple[float, float]:
    b = pd.read_csv(BAND_PATH)
    _must_have(b, ["lower_bound", "upper_bound"], "ohtani_attack_angle_band_rate.csv")
    lower = float(b["lower_bound"].iloc[0])
    upper = float(b["upper_bound"].iloc[0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"帯域が不正: lower={lower}, upper={upper}")
    return lower, upper


def _label_box(n_total: int, n_avail: int) -> str:
    miss = 1.0 - (n_avail / n_total) if n_total else float("nan")
    return f"n_total={n_total}\nattack_angle_available={n_avail}\nmissing_rate={miss:.3f}"


def main() -> None:
    df = pd.read_csv(CONTACT_PATH)
    _must_have(df, ["player_id", "strikes", "attack_angle"], "contact_quality.csv")

    lower, upper = _load_band()

    oht = df[df["player_id"] == "ohtani"].copy()
    if len(oht) == 0:
        raise ValueError("player_id=='ohtani' が0件です（表記を確認）")

    conds = [
        ("overall", oht),
        ("two_strike", oht[oht["strikes"] == 2]),
    ]

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True)
    bins = 60  # 視覚的に十分な解像度（固定）

    for ax, (name, d) in zip(axes, conds):
        n_total = int(len(d))
        aa = d["attack_angle"]
        aa_avail = aa.dropna()
        n_avail = int(len(aa_avail))

        # 固定帯域（半透明帯＋境界線）
        ax.axvspan(lower, upper, alpha=0.2)
        ax.axvline(lower, linestyle="--", linewidth=1.5)
        ax.axvline(upper, linestyle="--", linewidth=1.5)

        # ヒスト（欠損は分母から除外、件数で直感）
        ax.hist(aa_avail.to_numpy(dtype=float), bins=bins, alpha=0.85)

        ax.set_title(f"Ohtani attack_angle distribution | {name} (band fixed)")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)

        # 右上テキスト（防御材料）
        ax.text(
            0.98,
            0.98,
            _label_box(n_total, n_avail),
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

    axes[-1].set_xlabel("attack_angle (deg)")
    fig.suptitle("What is the fixed 'attack_angle band'? (Ohtani, overall vs two_strike)", y=0.98)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()


