from __future__ import annotations

"""
PhaseまとめMarkdown生成

出力:
- layer1_5/docs/phase_summary.md

ルール:
- 主張は1本
- 「リーグ平均」という語は禁止 → benchmark_group と書く
- 因果主張禁止（相関/差分/構造の記述まで）
"""

from pathlib import Path

import pandas as pd


OUT_MD = Path("docs/phase_summary.md")

BASE_TABLE = Path("data/output/phase_all_base_table.csv")
A0 = Path("data/output/A0_in_band_rate_check.csv")
A1 = Path("data/output/A1_out_of_band_penalty.csv")
A2 = Path("data/output/A2_bat_speed_effect_in_band.csv")
A3 = Path("data/output/A3_contact_efficiency_effect_in_band.csv")
A4 = Path("data/output/A4_two_strike_out_of_band_drivers.csv")


def md_table(df: pd.DataFrame, cols: list[str], float_cols: list[str] | None = None) -> str:
    d = df[cols].copy()
    float_cols = float_cols or []
    for c in float_cols:
        if c in d.columns:
            d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return d.to_markdown(index=False)


def main() -> None:
    base = pd.read_csv(BASE_TABLE)
    kpi_col = str(base["primary_kpi_source_col"].iloc[0]) if "primary_kpi_source_col" in base.columns else "primary_kpi"
    kpi_label = str(base["primary_kpi_label"].iloc[0]) if "primary_kpi_label" in base.columns else "primary_kpi"

    a0 = pd.read_csv(A0)
    a1 = pd.read_csv(A1)
    a2 = pd.read_csv(A2)
    a3 = pd.read_csv(A3)
    a4 = pd.read_csv(A4)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("## 1. 目的（1行）")
    lines.append("two_strike（strikes==2）で、attack_angleの固定帯域を外さない制御と、帯域内での出力を両輪として長打力向上を意思決定に落とす。")
    lines.append("")

    lines.append("## 2. 主張（1本）")
    lines.append("**長打を増やす鍵は「attack_angleの固定帯域を外さない（制御）」×「帯域内でEV/出力を上げる（出力）」であり、特にtwo_strikeでの再現性を最重要ターゲットとする。**")
    lines.append("")

    lines.append("## 3. 事実①：帯域内率（A0）")
    lines.append(md_table(a0, ["group", "condition", "n", "in_band_rate", "lower_bound", "upper_bound"], float_cols=["in_band_rate", "lower_bound", "upper_bound"]))
    lines.append("")

    lines.append("## 4. 事実②：帯域外ペナルティ（A1）")
    lines.append(f"Primary KPI: **{kpi_label}**（使用列: `{kpi_col}`）")
    lines.append(md_table(a1, ["group", "condition", "n_total", "n_in", "n_out", "mean_primary_kpi_in", "mean_primary_kpi_out", "diff_primary_kpi", "ratio_primary_kpi", "py1_in", "py1_out", "diff_py1"], float_cols=["mean_primary_kpi_in", "mean_primary_kpi_out", "diff_primary_kpi", "ratio_primary_kpi", "py1_in", "py1_out", "diff_py1"]))
    lines.append("")

    lines.append("## 5. 事実③：帯域内の出力レバー（A2）")
    lines.append("`in_band==True` のみで、bat_speedを分位化してKPIとP(y1=1)を確認。")
    lines.append(f"出力CSV: `{A2.as_posix()}` / 図: `data/output/plots/A2_bat_speed_effect_in_band.png`")
    lines.append("")

    lines.append("## 6. 事実④：帯域内の効率特徴（A3）")
    lines.append("`in_band==True` のみで、contact_efficiency（launch_speed/bat_speed）を分位化してKPIとP(y1=1)を確認。")
    lines.append("注意：これは相関の観察であり、因果は主張しない。")
    lines.append(f"出力CSV: `{A3.as_posix()}` / 図: `data/output/plots/A3_contact_efficiency_effect_in_band.png`")
    lines.append("")

    lines.append("## 7. 事実⑤：two_strike帯域外ドライバー（A4）")
    lines.append("two_strikeのみに限定し、out_band（帯域外）になりやすい特徴を分位×確率で確認。")
    lines.append(f"出力CSV: `{A4.as_posix()}` / 図: `data/output/plots/A4_two_strike_out_of_band_drivers.png`")
    lines.append("")

    lines.append("## 8. 提案（現場アクション：仮説ベースの介入）")
    lines.append("- **制御（攻角帯域）**: 固定帯域からの逸脱を減らす練習設計（two_strikeを優先窓にする）。")
    lines.append("- **出力（bat_speed）**: 帯域内に入った打球での出力上振れを狙う（帯域外に出ないことを前提条件にする）。")
    lines.append("- **インパクト条件（効率）**: contact_efficiencyが高い局面の特徴を、フォーム・タイミング・当て方の観察指標として使う（因果断定はしない）。")
    lines.append("- **two_strike特化ドリル**: A4で候補になった特徴（ブレの兆候）を減らす練習を優先（原因の断定はしない）。")
    lines.append("")

    lines.append("## 9. リスク/限界")
    lines.append("- データは上位打者に偏ったbenchmark_groupであり、一般化しない。")
    lines.append("- 単変量・分位×確率の観察であり、因果や最適値は主張しない。")
    lines.append("")

    lines.append("## 10. 次フェーズ")
    lines.append("- 球種・コース別の最適帯域は“補助”として検討（本筋に混ぜない）。")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {OUT_MD}")


if __name__ == "__main__":
    main()


