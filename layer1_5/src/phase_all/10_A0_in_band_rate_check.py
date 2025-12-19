from __future__ import annotations

"""
A0: 帯域内率の整合チェック

出力:
- data/output/A0_in_band_rate_check.csv
  columns: group, condition, n, in_band_rate, lower_bound, upper_bound

注意:
- 「リーグ平均」という語は使わず benchmark_group と表現する。
"""

from pathlib import Path

import pandas as pd

from ._utils_phase import Band, ensure_columns, read_band_from_single_source


BASE_TABLE = Path("data/output/phase_all_base_table.csv")
BAND_SOURCE = Path("data/output/ohtani_attack_angle_band_rate.csv")
OUT_CSV = Path("data/output/A0_in_band_rate_check.csv")

OHTANI_ID = "ohtani"


def rate(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return float("nan")
    return float(df["in_band"].mean())


def main() -> None:
    df = pd.read_csv(BASE_TABLE)
    ensure_columns(df, ["player_id", "two_strike", "in_band", "attack_angle"])
    band: Band = read_band_from_single_source(BAND_SOURCE)

    groups: dict[str, pd.DataFrame] = {
        "ohtani": df[df["player_id"] == OHTANI_ID],
        "benchmark_group": df,
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
                    "n": int(len(sdf)),
                    "in_band_rate": rate(sdf),
                    "lower_bound": band.lower,
                    "upper_bound": band.upper,
                }
            )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()


