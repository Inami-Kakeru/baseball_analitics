from __future__ import annotations

"""
in_band_rate 定義の完全一本化（単体タスク）

採用定義（唯一の正）:
- 分母: attack_angle が記録されている打球のみ（attack_angle非欠損）
- 帯域: data/output/ohtani_attack_angle_band_rate.csv の lower_bound/upper_bound（単一ソース）
- 計算: lower_bound <= attack_angle <= upper_bound
- 欠損: attack_angle欠損は分母から除外（out_band扱い禁止）

出力:
- data/output/in_band_rate_unified.csv
  columns: group, condition, n_total, in_band_rate
"""

from pathlib import Path

import pandas as pd

# 直接実行でも動くように絶対import（run_all_phase からも単体からもOK）
from layer1_5.src.phase_all._utils_phase import ensure_columns, read_band_from_single_source


BASE_TABLE = Path("data/output/phase_all_base_table.csv")
BAND_SOURCE = Path("data/output/ohtani_attack_angle_band_rate.csv")
OUT_CSV = Path("data/output/in_band_rate_unified.csv")

OHTANI_ID = "ohtani"


def compute(df: pd.DataFrame, lower: float, upper: float) -> float:
    if len(df) == 0:
        return float("nan")
    in_band = (df["attack_angle"] >= lower) & (df["attack_angle"] <= upper)
    return float(in_band.mean())


def main() -> None:
    df = pd.read_csv(BASE_TABLE)
    ensure_columns(df, ["player_id", "attack_angle", "strikes"])
    band = read_band_from_single_source(BAND_SOURCE)

    # 分母を attack_angle 非欠損に統一（欠損をout_band扱いしない）
    df_valid = df.dropna(subset=["attack_angle"]).copy()
    df_valid["two_strike"] = df_valid["strikes"] == 2

    groups: dict[str, pd.DataFrame] = {
        "ohtani": df_valid[df_valid["player_id"] == OHTANI_ID],
        "benchmark_group": df_valid,
    }

    rows = []
    for gname, gdf in groups.items():
        for cond, sdf in [
            ("overall", gdf),
            ("two_strike", gdf[gdf["two_strike"] == True]),  # noqa: E712
        ]:
            rows.append(
                {
                    "group": gname,
                    "condition": cond,
                    "n_total": int(len(sdf)),
                    "in_band_rate": compute(sdf, band.lower, band.upper),
                }
            )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()


