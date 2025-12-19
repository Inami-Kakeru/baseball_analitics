from __future__ import annotations

"""
最終提出用 図1〜5を既存CSVのみから再作成（分析の再実行・再学習なし）

出力（スライド番号と一致）:
- layer1_5/data/output/plots/final/slide06.png  (図1)
- layer1_5/data/output/plots/final/slide08.png  (図2)
- layer1_5/data/output/plots/final/slide09.png  (図3)
- layer1_5/data/output/plots/final/slide10.png  (図4)
- layer1_5/data/output/plots/final/slide11.png  (図5)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    # package 実行時
    from layer1_5.src.phase_C.final_viz_utils import setup_japanese_matplotlib, save_png_300dpi, style_axes
except ModuleNotFoundError:
    # script 実行時（python layer1_5/src/phase_C/make_submission_figures.py）
    from final_viz_utils import setup_japanese_matplotlib, save_png_300dpi, style_axes  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[3]
LAYER_DIR = REPO_ROOT / "layer1_5"
FINAL_DIR = LAYER_DIR / "data" / "output" / "plots" / "final"

# 単一ソース（唯一の正）
BAND_PATH = LAYER_DIR / "data" / "output" / "ohtani_attack_angle_band_rate.csv"

# 既存CSV
CONTACT_QUALITY = LAYER_DIR / "data" / "intermediate" / "contact_quality.csv"
C1 = LAYER_DIR / "data" / "output" / "C1_twostrike_band_penalty.csv"
C2 = LAYER_DIR / "data" / "output" / "C2_bandwidth_sensitivity.csv"
C4 = LAYER_DIR / "data" / "output" / "C4_multi_kpi_band_penalty.csv"
C6 = LAYER_DIR / "data" / "output" / "C6_ohtani_action_diagnosis_summary.csv"


def must_have(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} 必須列が不足: {missing}")


def load_band() -> tuple[float, float]:
    b = pd.read_csv(BAND_PATH)
    must_have(b, ["lower_bound", "upper_bound"], "ohtani_attack_angle_band_rate.csv")
    lower = float(b["lower_bound"].iloc[0])
    upper = float(b["upper_bound"].iloc[0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"帯域が不正: lower={lower}, upper={upper}")
    return lower, upper


def fig1_attack_angle_distribution() -> Path:
    """
    図1: attack_angle 分布と固定帯域（概念理解用）
    """
    lower, upper = load_band()
    df = pd.read_csv(CONTACT_QUALITY)
    must_have(df, ["player_id", "strikes", "attack_angle"], "contact_quality.csv")

    oht = df[df["player_id"] == "ohtani"].copy()
    if len(oht) == 0:
        raise ValueError("player_id=='ohtani' が0件です")

    def box_text(d: pd.DataFrame) -> str:
        n_total = int(len(d))
        n_avail = int(d["attack_angle"].notna().sum())
        miss = 1.0 - (n_avail / n_total) if n_total else float("nan")
        return f"n_total={n_total}\nattack_angle非欠損={n_avail}\n欠損率={miss:.3f}"

    setup_japanese_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.2), sharex=True)

    bins = 60
    for ax, (cond, d) in zip(
        axes,
        [
            ("overall", oht),
            ("two_strike（2スト）", oht[oht["strikes"] == 2]),
        ],
    ):
        aa = d["attack_angle"].dropna().to_numpy(dtype=float)
        ax.hist(aa, bins=bins, color="#4C78A8", alpha=0.85)
        ax.axvspan(lower, upper, color="#999999", alpha=0.25)
        ax.axvline(lower, color="#666666", linestyle="--", linewidth=1.2)
        ax.axvline(upper, color="#666666", linestyle="--", linewidth=1.2)
        ax.set_ylabel("件数")
        style_axes(ax)
        ax.set_title(f"{cond} の攻角（attack_angle）分布（灰色帯＝固定帯域）")
        ax.text(
            0.98,
            0.98,
            box_text(d),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )

    axes[-1].set_xlabel("攻角（attack_angle, 度）")
    fig.suptitle("攻角（attack_angle）分布と固定帯域（大谷翔平）— 帯域は再学習せず事前固定", y=0.98, fontsize=14)

    # フッター（共通ルール）
    footer = (
        "注記：灰色帯＝事前に固定した攻角帯域（単一ソース：ohtani_attack_angle_band_rate.csv、再学習なし）／"
        "分母＝攻角（attack_angle）非欠損のみ（欠損は帯域外扱いしない）"
    )
    axes[-1].text(0, -0.22, footer, transform=axes[-1].transAxes, ha="left", va="top", fontsize=10)

    out = FINAL_DIR / "slide06.png"
    save_png_300dpi(out)
    plt.close()
    return out


def fig2_twostrike_inout_bar() -> Path:
    """
    図2: two_strike 内の in_band / out_band 比較（核）
    """
    df = pd.read_csv(C1)
    must_have(
        df,
        ["group", "n_total", "n_attack_angle_available", "n_in", "n_out", "mean_xslg_in", "mean_xslg_out", "diff_xslg"],
        "C1_twostrike_band_penalty.csv",
    )

    oht = df[df["group"] == "ohtani"].iloc[0].to_dict()
    bench = df[df["group"] == "benchmark_group"].iloc[0].to_dict()

    setup_japanese_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    def draw(ax: plt.Axes, rec: dict, title: str) -> None:
        vals = [rec["mean_xslg_in"], rec["mean_xslg_out"]]
        labels = ["帯域内（in_band）", "帯域外（out_band）"]
        colors = ["#4C78A8", "#F58518"]
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors)
        style_axes(ax)
        ax.set_title(title)
        ax.set_ylabel("平均 xSLG（推定SLG）")
        dx = float(rec["diff_xslg"])
        ax.text(
            0.5,
            0.95,
            f"ΔxSLG（内−外）={dx:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )
        # n の明記
        nbox = (
            f"条件：two_strike（2スト）\n"
            f"n_total={int(rec['n_total'])}\n"
            f"分母（攻角attack_angle非欠損）={int(rec['n_attack_angle_available'])}\n"
            f"内={int(rec['n_in'])} / 外={int(rec['n_out'])}"
        )
        ax.text(
            0.98,
            0.02,
            nbox,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    draw(axes[0], oht, "大谷翔平（2スト）: 帯域内の方が高いxSLG")
    draw(axes[1], bench, "上位打者サンプル（2スト）: 帯域内の方が高いxSLG")

    fig.suptitle("two_strike 内でも「帯域外」は追加的損失と関連（観察）", y=1.02, fontsize=14)
    footer = "注記：『2ストで落ちる』こと自体ではなく、落ちた世界の中での追加損失（因果は主張しない）"
    axes[0].text(0, -0.22, footer, transform=axes[0].transAxes, ha="left", va="top", fontsize=10)

    out = FINAL_DIR / "slide08.png"
    save_png_300dpi(out)
    plt.close()
    return out


def fig3_bandwidth_sensitivity_twostrike() -> Path:
    """
    図3: 帯域幅感度分析（防御用）※two_strikeのみ
    """
    df = pd.read_csv(C2)
    must_have(df, ["band", "condition", "group", "diff_xslg", "n_in", "n_out"], "C2_bandwidth_sensitivity.csv")
    sub = df[df["condition"] == "two_strike"].copy()

    order = ["narrow6", "narrow4", "narrow2", "base", "wide2", "wide4", "wide6"]
    label_map = {
        "narrow6": "狭い（±6°）",
        "narrow4": "狭い（±4°）",
        "narrow2": "狭い（±2°）",
        "base": "基準（±0°）",
        "wide2": "広い（±2°）",
        "wide4": "広い（±4°）",
        "wide6": "広い（±6°）",
    }
    sub = sub[sub["band"].isin(order)]
    sub["band_order"] = sub["band"].map({k: i for i, k in enumerate(order)})
    sub = sub.sort_values("band_order")

    setup_japanese_matplotlib()
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    style_axes(ax)

    for g, color in [("ohtani", "#4C78A8"), ("benchmark_group", "#F58518")]:
        gdf = sub[sub["group"] == g].copy()
        x = [label_map[b] for b in gdf["band"].tolist()]
        y = gdf["diff_xslg"].to_numpy(dtype=float)
        # サンプル崩壊目安（どちらかが少ない）
        collapse = (gdf["n_in"] < 30) | (gdf["n_out"] < 30)
        ax.plot(x, y, marker="o", color=color, label=("大谷翔平" if g == "ohtani" else "上位打者サンプル"))
        # 崩壊点を強調
        if collapse.any():
            ax.scatter(np.array(x)[collapse.to_numpy()], y[collapse.to_numpy()], s=90, facecolors="none", edgecolors="red", linewidths=2, zorder=3)

    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_ylabel("ΔxSLG（帯域内 − 帯域外）")
    ax.set_xlabel("帯域幅設定（固定帯域を±で拡張/縮小）")
    ax.set_title("合理的な帯域幅では結論の符号は維持される（2スト）")
    ax.legend(loc="upper left")

    footer = (
        "注記：条件＝two_strike（2スト）／分母＝攻角（attack_angle）非欠損のみ／"
        "赤枠＝内/外のどちらかが n<30 でサンプル崩壊（検証不能に近い）"
    )
    ax.text(0, -0.22, footer, transform=ax.transAxes, ha="left", va="top", fontsize=10)

    out = FINAL_DIR / "slide09.png"
    save_png_300dpi(out)
    plt.close()
    return out


def fig4_kpi_crosscheck_twostrike() -> Path:
    """
    図4: KPI横断（xSLG / xwOBA）※two_strikeのみ
    """
    df = pd.read_csv(C4)
    must_have(df, ["condition", "group", "kpi", "diff", "n_attack_angle_available", "n_in", "n_out"], "C4_multi_kpi_band_penalty.csv")
    sub = df[df["condition"] == "two_strike"].copy()

    # 利用可能KPIのみ
    kpi_map = {
        "estimated_slg_using_speedangle": "xSLG（推定SLG）",
        "estimated_woba_using_speedangle": "xwOBA（推定wOBA）",
        "barrel": "Barrel率",
        "hard_hit": "HardHit率",
    }
    sub["kpi_jp"] = sub["kpi"].map(lambda x: kpi_map.get(x, str(x)))
    kpis = [k for k in ["estimated_slg_using_speedangle", "estimated_woba_using_speedangle", "barrel", "hard_hit"] if k in sub["kpi"].unique()]
    sub = sub[sub["kpi"].isin(kpis)]
    if len(kpis) == 0:
        raise ValueError("two_strikeで利用可能なKPIが見つかりません")

    setup_japanese_matplotlib()
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    style_axes(ax)

    x_labels = [kpi_map.get(k, k) for k in kpis]
    x = np.arange(len(x_labels))
    width = 0.36

    def series(group: str) -> np.ndarray:
        vals = []
        for k in kpis:
            r = sub[(sub["group"] == group) & (sub["kpi"] == k)]
            vals.append(float(r["diff"].iloc[0]) if len(r) else float("nan"))
        return np.array(vals, dtype=float)

    y_oht = series("ohtani")
    y_ben = series("benchmark_group")

    ax.bar(x - width / 2, y_oht, width, label="大谷翔平", color="#4C78A8")
    ax.bar(x + width / 2, y_ben, width, label="上位打者サンプル", color="#F58518")
    ax.axhline(0, color="#333333", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Δ（帯域内 − 帯域外）")
    ax.set_title("KPIを変えても in/out の関係は同方向（2スト・利用可能範囲）")
    ax.legend(loc="upper left")

    footer = "注記：条件＝two_strike（2スト）／分母＝攻角（attack_angle）非欠損のみ／利用可能なKPI範囲で比較（欠損KPIは未使用）"
    ax.text(0, -0.22, footer, transform=ax.transAxes, ha="left", va="top", fontsize=10)

    out = FINAL_DIR / "slide10.png"
    save_png_300dpi(out)
    plt.close()
    return out


def fig5_ohtani_diagnosis_summary() -> Path:
    """
    図5: 大谷翔平の診断まとめ
    - in_band_rate（overall / two_strike）
    - ΔxSLG（overall / two_strike）
    """
    df = pd.read_csv(C6)
    must_have(df, ["condition", "n_total", "n_attack_angle_available", "in_band_rate", "diff_xslg"], "C6_ohtani_action_diagnosis_summary.csv")

    # 条件順
    order = ["overall", "two_strike"]
    d = df.copy()
    d["condition_order"] = d["condition"].map({k: i for i, k in enumerate(order)})
    d = d.sort_values("condition_order")

    cond_labels = ["overall", "two_strike（2スト）"]
    in_rate = d["in_band_rate"].to_numpy(dtype=float)
    dx = d["diff_xslg"].to_numpy(dtype=float)

    # 欠損率（分母定義の明示に使う）
    n_total = d["n_total"].to_numpy(dtype=int)
    n_avail = d["n_attack_angle_available"].to_numpy(dtype=int)
    miss = 1.0 - (n_avail / n_total)

    setup_japanese_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # 左：in_band_rate
    ax = axes[0]
    style_axes(ax)
    ax.bar(cond_labels, in_rate, color="#4C78A8", alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("帯域内率（分母＝攻角attack_angle非欠損）")
    ax.set_title("帯域集中度（in_band_rate）は大きく崩れない")
    for i, v in enumerate(in_rate):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
        ax.text(i, 0.02, f"n={n_avail[i]}（欠損率{miss[i]:.3f}）", ha="center", va="bottom", fontsize=10)

    # 右：ΔxSLG
    ax = axes[1]
    style_axes(ax)
    ax.bar(cond_labels, dx, color="#F58518", alpha=0.9)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_ylabel("ΔxSLG（帯域内 − 帯域外）")
    ax.set_title("帯域外の損失（ΔxSLG）は2ストでも大きい")
    ax.set_ylim(0, float(np.nanmax(dx)) * 1.25)
    for i, v in enumerate(dx):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
        ax.text(i, min(0.02, v * 0 + 0.02), f"n={n_avail[i]}（分母）", ha="center", va="bottom", fontsize=10)

    fig.suptitle("大谷翔平：帯域集中は維持、帯域外の損失が課題（観察）", y=1.02, fontsize=14)
    footer = "注記：『当てに行くこと』自体を否定しない。介入対象は two_strike（2スト）での帯域外（out_band）発生（崩れ方の低減）"
    axes[0].text(0, -0.22, footer, transform=axes[0].transAxes, ha="left", va="top", fontsize=10)

    out = FINAL_DIR / "slide11.png"
    save_png_300dpi(out)
    plt.close()
    return out


def main() -> None:
    # どこから実行しても安定するように、cwd を layer1_5 に寄せる
    os.chdir(LAYER_DIR)
    # script実行時の相対import用
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    outs = [
        fig1_attack_angle_distribution(),
        fig2_twostrike_inout_bar(),
        fig3_bandwidth_sensitivity_twostrike(),
        fig4_kpi_crosscheck_twostrike(),
        fig5_ohtani_diagnosis_summary(),
    ]
    print("\n=== saved (final submission figures) ===")
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()


