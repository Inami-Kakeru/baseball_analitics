from __future__ import annotations

"""
PPT差し替え用: 全グラフを「日本語表記・統一デザイン・16:9最適（1920×1080）」
既存CSVのみを使用（分析の再実行・再学習なし）。Seaborn禁止（matplotlibのみ）。

出力先: layer1_5/data/output/plots/rev/
出力ファイル（固定）:
 (1) rev_fig06_attack_angle_hist_band.png
 (2) rev_fig08_inout_mean_xslg_twostrike.png
 (3) rev_fig09_distance_vs_xslg_twostrike.png
 (4) rev_fig09_out_low_high_asymmetry.png
 (5) rev_appx_bandwidth_sensitivity.png
 (6) rev_appx_kpi_robustness.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from layer1_5.src.phase_C.rev_viz_utils import (
    FigSpec,
    NAVY,
    ORANGE,
    GRAY,
    BLACK,
    setup_fonts,
    style_axes,
    add_note_bottom_right,
    save_png,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LAYER_DIR = REPO_ROOT / "layer1_5"
OUT_DIR = LAYER_DIR / "data" / "output" / "plots" / "rev"

# 入力（既存）
CONTACT = LAYER_DIR / "data" / "intermediate" / "contact_quality.csv"
BAND = LAYER_DIR / "data" / "output" / "ohtani_attack_angle_band_rate.csv"
C1 = LAYER_DIR / "data" / "output" / "C1_twostrike_band_penalty.csv"
C2 = LAYER_DIR / "data" / "output" / "C2_bandwidth_sensitivity.csv"
C4 = LAYER_DIR / "data" / "output" / "C4_multi_kpi_band_penalty.csv"
C6 = LAYER_DIR / "data" / "output" / "C6_ohtani_action_diagnosis_summary.csv"
APPX_SIDE = LAYER_DIR / "data" / "output" / "appendix_outband_side_delta.csv"


def must_have(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} 必須列が不足: {missing}")


def load_band() -> tuple[float, float, float]:
    b = pd.read_csv(BAND)
    must_have(b, ["lower_bound", "upper_bound"], "ohtani_attack_angle_band_rate.csv")
    lower = float(b["lower_bound"].iloc[0])
    upper = float(b["upper_bound"].iloc[0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"帯域が不正: lower={lower}, upper={upper}")
    center = (lower + upper) / 2.0
    return lower, upper, center


def denom_note(condition: str, n_total: int, n_avail: int, lower: float, upper: float) -> str:
    miss = 1.0 - (n_avail / n_total) if n_total else float("nan")
    return (
        f"条件：{condition} / 分母：attack_angle非欠損のみ（欠損は帯域外扱いしない） / "
        f"帯域：単一ソース固定（再学習なし、lower={lower:.2f}°, upper={upper:.2f}°） / "
        f"n_total={n_total}, n={n_avail}, 欠損率={miss:.3f}"
    )


def fig1_hist_band(spec: FigSpec) -> Path:
    lower, upper, _ = load_band()
    df = pd.read_csv(CONTACT)
    must_have(df, ["player_id", "strikes", "attack_angle"], "contact_quality.csv")
    oht = df[df["player_id"] == "ohtani"].copy()
    if len(oht) == 0:
        raise ValueError("player_id=='ohtani' が0件です")

    def panel(ax: plt.Axes, d: pd.DataFrame, cond: str) -> None:
        n_total = int(len(d))
        aa = d["attack_angle"]
        aa_av = aa.dropna().to_numpy(dtype=float)
        n_av = int(len(aa_av))
        ax.hist(aa_av, bins=60, color=NAVY, alpha=0.85)
        ax.axvspan(lower, upper, color=GRAY, alpha=0.25)
        ax.axvline(lower, color=GRAY, linestyle="--", linewidth=1.3)
        ax.axvline(upper, color=GRAY, linestyle="--", linewidth=1.3)
        ax.set_title(f"{cond} の攻角（attack_angle）分布：灰色帯＝固定帯域", fontsize=14)
        ax.set_ylabel("件数")
        style_axes(ax)
        # 右上にn
        miss = 1.0 - (n_av / n_total) if n_total else float("nan")
        ax.text(
            0.98,
            0.98,
            f"n_total={n_total}\n非欠損n={n_av}\n欠損率={miss:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )

    fig, axes = plt.subplots(2, 1, figsize=spec.figsize, dpi=spec.dpi, sharex=True)
    panel(axes[0], oht, "overall")
    panel(axes[1], oht[oht["strikes"] == 2], "two_strike（2スト）")
    axes[1].set_xlabel("攻角（attack_angle）[度]")
    fig.suptitle("帯域は“勝手に作っていない”：大谷の攻角分布に固定帯域を重ねて可視化", y=0.98, fontsize=18)

    # 右下注記（統一フォーマット）
    n_total = int(len(oht[oht["strikes"] == 2]))
    n_av = int(oht[oht["strikes"] == 2]["attack_angle"].notna().sum())
    add_note_bottom_right(axes[1], denom_note("overall / two_strike", n_total=n_total, n_avail=n_av, lower=lower, upper=upper))

    out = OUT_DIR / "rev_fig06_attack_angle_hist_band.png"
    save_png(out, dpi=spec.dpi)
    plt.close()
    return out


def fig2_inout_xslg_twostrike(spec: FigSpec) -> Path:
    lower, upper, _ = load_band()
    df = pd.read_csv(C1)
    must_have(df, ["group", "n_total", "n_attack_angle_available", "mean_xslg_in", "mean_xslg_out", "diff_xslg", "n_in", "n_out"], "C1_twostrike_band_penalty.csv")
    oht = df[df["group"] == "ohtani"].iloc[0].to_dict()
    ben = df[df["group"] == "benchmark_group"].iloc[0].to_dict()

    def panel(ax: plt.Axes, rec: dict, title: str) -> None:
        x = np.arange(2)
        vals = [float(rec["mean_xslg_in"]), float(rec["mean_xslg_out"])]
        ax.bar(x, vals, color=[NAVY, ORANGE], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(["帯域内（in_band）", "帯域外（out_band）"])
        ax.set_ylabel("平均 xSLG（推定SLG）")
        ax.set_title(title, fontsize=14)
        style_axes(ax)
        dx = float(rec["diff_xslg"])
        ax.text(0.5, 0.92, f"ΔxSLG（内−外）={dx:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=18, color=BLACK,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
        ax.text(
            0.98,
            0.02,
            f"n_total={int(rec['n_total'])}\n分母n={int(rec['n_attack_angle_available'])}\n内={int(rec['n_in'])} / 外={int(rec['n_out'])}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )

    fig, axes = plt.subplots(1, 2, figsize=spec.figsize, dpi=spec.dpi, sharey=True)
    panel(axes[0], oht, "大谷翔平（2スト）")
    panel(axes[1], ben, "上位打者サンプル（2スト）")
    fig.suptitle("2ストの“落ちた世界”でも、帯域外は追加的損失と関連（観察）", y=0.98, fontsize=18)
    add_note_bottom_right(
        axes[1],
        denom_note("two_strike（2スト）", n_total=int(oht["n_total"]), n_avail=int(oht["n_attack_angle_available"]), lower=lower, upper=upper),
    )

    out = OUT_DIR / "rev_fig08_inout_mean_xslg_twostrike.png"
    save_png(out, dpi=spec.dpi)
    plt.close()
    return out


def moving_avg_curve(d: pd.DataFrame, bin_deg: float = 0.5, min_n: int = 20, roll: int = 7) -> pd.DataFrame:
    x = d["dist"].to_numpy(dtype=float)
    hi = float(np.ceil(np.nanmax(x) / bin_deg) * bin_deg)
    edges = np.arange(0.0, hi + bin_deg, bin_deg)
    if len(edges) < 3:
        return pd.DataFrame()
    tmp = d.copy()
    tmp["bin"] = pd.cut(tmp["dist"], bins=edges, include_lowest=True)
    g = (
        tmp.groupby("bin", dropna=True)
        .agg(n=("xslg", "count"), mean=("xslg", "mean"), x_mean=("dist", "mean"))
        .reset_index(drop=True)
        .sort_values("x_mean")
    )
    g = g[g["n"] >= min_n].copy()
    if len(g) == 0:
        return pd.DataFrame()
    g["smooth"] = g["mean"].rolling(roll, center=True, min_periods=max(2, roll // 2)).mean()
    return g


def fig3_distance_vs_xslg(spec: FigSpec) -> Path:
    lower, upper, center = load_band()
    df = pd.read_csv(CONTACT)
    must_have(df, ["player_id", "strikes", "attack_angle", "estimated_slg_using_speedangle"], "contact_quality.csv")

    oht2 = df[(df["player_id"] == "ohtani") & (df["strikes"] == 2)].copy()
    n_total = int(len(oht2))
    d = oht2.dropna(subset=["attack_angle", "estimated_slg_using_speedangle"]).copy()
    n_av = int(len(d))
    if n_av == 0:
        raise ValueError("two_strikeで attack_angle/xSLG が揃う行が0です")

    d["xslg"] = d["estimated_slg_using_speedangle"].astype(float)
    d["dist"] = (d["attack_angle"].astype(float) - center).abs()
    d["in_band"] = (d["attack_angle"] >= lower) & (d["attack_angle"] <= upper)
    in_d = d[d["in_band"] == True]  # noqa: E712
    out_d = d[d["in_band"] == False]  # noqa: E712
    curve = moving_avg_curve(d, bin_deg=0.5, min_n=20, roll=7)

    fig, ax = plt.subplots(1, 1, figsize=spec.figsize, dpi=spec.dpi)
    style_axes(ax)
    ax.scatter(in_d["dist"], in_d["xslg"], s=22, alpha=0.35, color=NAVY, label="帯域内（in_band）")
    ax.scatter(out_d["dist"], out_d["xslg"], s=22, alpha=0.25, color=ORANGE, label="帯域外（out_band）")
    if len(curve):
        ax.plot(curve["x_mean"], curve["smooth"], color=BLACK, linewidth=3.0, label="距離ビン0.5°の移動平均（rolling）")

    ax.set_xlabel("|攻角（attack_angle）− 帯域中心| [度]")
    ax.set_ylabel("xSLG（推定SLG）")
    ax.set_title("帯域中心から離れるほど xSLG は“連続的に”低下しやすい（大谷・2スト）", fontsize=18)
    ax.legend(loc="upper right")

    # 図内に帯域中心/境界
    ax.text(
        0.02,
        0.95,
        f"帯域中心={center:.2f}°（lower={lower:.2f}°, upper={upper:.2f}°）",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
    )

    add_note_bottom_right(ax, denom_note("two_strike（2スト）", n_total=n_total, n_avail=n_av, lower=lower, upper=upper))

    out = OUT_DIR / "rev_fig09_distance_vs_xslg_twostrike.png"
    save_png(out, dpi=spec.dpi)
    plt.close()
    return out


def fig4_low_high_asymmetry(spec: FigSpec) -> Path:
    lower, upper, _ = load_band()
    df = pd.read_csv(APPX_SIDE)
    must_have(df, ["condition", "side", "diff_from_in", "n"], "appendix_outband_side_delta.csv")

    # 可読性優先：2条件（all と EV>=95）を2本で表示（情報過多なら後で1条件へ）
    cond_order = ["two_strike_all", "two_strike_launch_speed>=95"]
    side_order = ["in_band", "out_low", "out_high"]
    side_label = {"in_band": "帯域内", "out_low": "帯域外（低側）", "out_high": "帯域外（高側）"}
    cond_label = {"two_strike_all": "2スト（全体）", "two_strike_launch_speed>=95": "2スト＋EV≥95mph"}

    sub = df[df["condition"].isin(cond_order) & df["side"].isin(side_order)].copy()
    sub["side_order"] = sub["side"].map({k: i for i, k in enumerate(side_order)})
    sub["cond_order"] = sub["condition"].map({k: i for i, k in enumerate(cond_order)})
    sub = sub.sort_values(["side_order", "cond_order"])

    fig, ax = plt.subplots(1, 1, figsize=spec.figsize, dpi=spec.dpi)
    style_axes(ax)

    x = np.arange(len(side_order))
    width = 0.35

    def yvals(cond: str) -> np.ndarray:
        vals = []
        for s in side_order:
            r = sub[(sub["condition"] == cond) & (sub["side"] == s)]
            vals.append(float(r["diff_from_in"].iloc[0]) if len(r) else float("nan"))
        return np.array(vals, dtype=float)

    y1 = yvals(cond_order[0])
    y2 = yvals(cond_order[1])

    ax.bar(x - width / 2, y1, width, color=NAVY, alpha=0.9, label=cond_label[cond_order[0]])
    ax.bar(x + width / 2, y2, width, color=ORANGE, alpha=0.9, label=cond_label[cond_order[1]])

    ax.set_xticks(x)
    ax.set_xticklabels([side_label[s] for s in side_order])
    ax.set_ylabel("ΔxSLG（帯域内 − 対象）")
    ax.set_title("帯域外の“低側/高側”で損失は非対称か？（大谷・2スト）", fontsize=18)
    ax.legend(loc="upper right")

    add_note_bottom_right(
        ax,
        f"条件：two_strike（2スト） / 分母：attack_angle非欠損のみ（欠損は帯域外扱いしない） / "
        f"帯域：単一ソース固定（再学習なし、lower={lower:.2f}°, upper={upper:.2f}°） / "
        "EV固定：launch_speed>=95mph（利用可能範囲で再確認）",
    )

    out = OUT_DIR / "rev_fig09_out_low_high_asymmetry.png"
    save_png(out, dpi=spec.dpi)
    plt.close()
    return out


def fig5_bandwidth_sensitivity(spec: FigSpec) -> Path:
    lower, upper, _ = load_band()
    df = pd.read_csv(C2)
    must_have(df, ["band", "condition", "group", "diff_xslg", "n_in", "n_out"], "C2_bandwidth_sensitivity.csv")
    sub = df[df["condition"] == "two_strike"].copy()

    order = ["narrow6", "narrow4", "narrow2", "base", "wide2", "wide4", "wide6"]
    label = {
        "narrow6": "狭い（±6°）",
        "narrow4": "狭い（±4°）",
        "narrow2": "狭い（±2°）",
        "base": "基準（±0°）",
        "wide2": "広い（±2°）",
        "wide4": "広い（±4°）",
        "wide6": "広い（±6°）",
    }
    sub = sub[sub["band"].isin(order)].copy()
    sub["order"] = sub["band"].map({k: i for i, k in enumerate(order)})
    sub = sub.sort_values("order")

    fig, ax = plt.subplots(1, 1, figsize=spec.figsize, dpi=spec.dpi)
    style_axes(ax)

    for g, color, gname in [("ohtani", NAVY, "大谷翔平"), ("benchmark_group", ORANGE, "上位打者サンプル")]:
        gdf = sub[sub["group"] == g].copy()
        x = np.arange(len(gdf))
        y = gdf["diff_xslg"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.8, color=color, label=gname)
        collapse = (gdf["n_in"] < 30) | (gdf["n_out"] < 30)
        if collapse.any():
            ax.scatter(x[collapse.to_numpy()], y[collapse.to_numpy()], s=140, facecolors="none", edgecolors="red", linewidths=2.5, zorder=3)

    ax.axhline(0, color=BLACK, linewidth=1.2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([label[b] for b in order])
    ax.set_ylabel("ΔxSLG（帯域内 − 帯域外）")
    ax.set_xlabel("帯域幅設定（固定帯域を拡張/縮小）")
    ax.set_title("合理的な帯域幅では結論の符号は維持される（2スト・付録）", fontsize=18)
    ax.legend(loc="upper left")

    add_note_bottom_right(
        ax,
        f"条件：two_strike（2スト） / 分母：attack_angle非欠損のみ（欠損は帯域外扱いしない） / "
        f"帯域：単一ソース固定（再学習なし、lower={lower:.2f}°, upper={upper:.2f}°） / "
        "赤丸：n<30でサンプル崩壊（検証不能に近い）",
    )

    out = OUT_DIR / "rev_appx_bandwidth_sensitivity.png"
    save_png(out, dpi=spec.dpi)
    plt.close()
    return out


def fig6_kpi_robustness(spec: FigSpec) -> Path:
    lower, upper, _ = load_band()
    df = pd.read_csv(C4)
    must_have(df, ["condition", "group", "kpi", "diff"], "C4_multi_kpi_band_penalty.csv")
    sub = df[df["condition"] == "two_strike"].copy()

    # 利用可能なKPIのうち、要求の2つを優先
    kpis = [k for k in ["estimated_slg_using_speedangle", "estimated_woba_using_speedangle"] if k in sub["kpi"].unique()]
    if len(kpis) == 0:
        raise ValueError("two_strikeで利用可能なKPIがありません")
    kpi_label = {
        "estimated_slg_using_speedangle": "ΔxSLG（推定SLG）",
        "estimated_woba_using_speedangle": "ΔxwOBA（推定wOBA）",
    }
    xlab = [kpi_label.get(k, k) for k in kpis]
    x = np.arange(len(kpis))
    width = 0.36

    def series(group: str) -> np.ndarray:
        vals = []
        for k in kpis:
            r = sub[(sub["group"] == group) & (sub["kpi"] == k)]
            vals.append(float(r["diff"].iloc[0]) if len(r) else float("nan"))
        return np.array(vals, dtype=float)

    y_oht = series("ohtani")
    y_ben = series("benchmark_group")

    fig, ax = plt.subplots(1, 1, figsize=spec.figsize, dpi=spec.dpi)
    style_axes(ax)
    ax.bar(x - width / 2, y_oht, width, color=NAVY, alpha=0.9, label="大谷翔平")
    ax.bar(x + width / 2, y_ben, width, color=ORANGE, alpha=0.9, label="上位打者サンプル")
    ax.axhline(0, color=BLACK, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(xlab)
    ax.set_ylabel("Δ（帯域内 − 帯域外）")
    ax.set_title("KPIを変えても in/out の関係は同方向（2スト・付録）", fontsize=18)
    ax.legend(loc="upper left")

    add_note_bottom_right(
        ax,
        f"条件：two_strike（2スト） / 分母：attack_angle非欠損のみ（欠損は帯域外扱いしない） / "
        f"帯域：単一ソース固定（再学習なし、lower={lower:.2f}°, upper={upper:.2f}°） / "
        "利用可能なKPI範囲で比較（欠損KPIは未使用）",
    )

    out = OUT_DIR / "rev_appx_kpi_robustness.png"
    save_png(out, dpi=spec.dpi)
    plt.close()
    return out


def main() -> None:
    os.chdir(LAYER_DIR)
    # font設定（ログに残す）
    chosen = setup_fonts()
    print(f"[FONT] chosen={chosen}")

    spec = FigSpec(width_px=1920, height_px=1080, dpi=200)
    outs = [
        fig1_hist_band(spec),
        fig2_inout_xslg_twostrike(spec),
        fig3_distance_vs_xslg(spec),
        fig4_low_high_asymmetry(spec),
        fig5_bandwidth_sensitivity(spec),
        fig6_kpi_robustness(spec),
    ]

    print("\n=== saved (rev figures) ===")
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()


