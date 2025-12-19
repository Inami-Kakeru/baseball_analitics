from __future__ import annotations

"""
phase_C_summary.md 生成

必須要点:
- two_strike“内”でも in_band vs out_band の差が残る（C1）
- 帯域幅を揺らしても差が残る（C2）
- 介入対象 = out_band を減らす（崩れ方の低減）
- 成功指標例: two_strike diff_xslg の縮小 / out_band率の低下
- benchmark_group は上位打者サンプルで一般化しない
- 因果主張しない（単変量観察）
"""

from pathlib import Path

import pandas as pd


OUT_MD = Path("docs/phase_C_summary.md")

C1 = Path("data/output/C1_twostrike_band_penalty.csv")
C2 = Path("data/output/C2_bandwidth_sensitivity.csv")
C3 = Path("data/output/C3_twostrike_outband_drivers.csv")
C4 = Path("data/output/C4_multi_kpi_band_penalty.csv")
C5 = Path("data/output/C5_shape_by_group.csv")
C6_SUM = Path("data/output/C6_ohtani_action_diagnosis_summary.csv")
C6_DRV = Path("data/output/C6_ohtani_twostrike_outband_drivers.csv")


def md_table(df: pd.DataFrame, cols: list[str], float_cols: list[str] | None = None) -> str:
    d = df[cols].copy()
    float_cols = float_cols or []
    for c in float_cols:
        if c in d.columns:
            d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return d.to_markdown(index=False)


def main() -> None:
    c1 = pd.read_csv(C1)
    c2 = pd.read_csv(C2)
    c4 = pd.read_csv(C4)
    c6s = pd.read_csv(C6_SUM)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("## 目的")
    lines.append("two_strike（strikes==2）で落ちるのは前提として認めた上で、その“中で”帯域外（out_band）が追加の損失を生むかを確認し、意思決定に使える介入対象へ接続する。")
    lines.append("")

    lines.append("## 主要な答え（短く）")
    lines.append("- two_strike“内”でも、in_band vs out_band で xSLG と P(y1=1) の差が残る（C1）。")
    lines.append("- 帯域幅を±2/±4/±6度で揺らしても、差が大きく崩れないかを確認できる（C2）。")
    lines.append("")

    lines.append("## C1: two_strike band penalty（主張の芯）")
    lines.append(md_table(c1, ["group", "n_attack_angle_available", "n_in", "n_out", "diff_xslg", "ratio_xslg", "diff_py1"], float_cols=["diff_xslg", "ratio_xslg", "diff_py1"]))
    lines.append("")

    lines.append("## C2: bandwidth sensitivity（恣意性の排除）")
    # base と wide/narrow のdiff_xslgだけ抜粋（two_strike）
    c2_ts = c2[c2["condition"] == "two_strike"][["band", "group", "diff_xslg"]].copy()
    lines.append(md_table(c2_ts, ["band", "group", "diff_xslg"], float_cols=["diff_xslg"]))
    lines.append("")

    lines.append("## 提案への接続（因果ではなく、仮説ベースの介入）")
    lines.append("- 介入対象は『2ストで out_band を減らす（崩れ方の低減）』。")
    lines.append("- 成功指標（例）: two_strikeでの `diff_xslg` の縮小、`P(out_band=1)` の低下。")
    lines.append("- benchmark_group は上位打者サンプルであり、一般化しない。")
    lines.append("- 単変量観察の結果であり、因果は主張しない。")
    lines.append("")

    lines.append("## 補助（幅の納得: KPI複数）")
    lines.append("出力CSV: `data/output/C4_multi_kpi_band_penalty.csv`（図は作らない）")
    lines.append("")

    lines.append("## 大谷のみ診断（改善プラン直結の材料）")
    lines.append(md_table(c6s, ["condition", "n_attack_angle_available", "in_band_rate", "diff_xslg", "diff_py1"], float_cols=["in_band_rate", "diff_xslg", "diff_py1"]))
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {OUT_MD}")


if __name__ == "__main__":
    main()


