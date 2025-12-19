from __future__ import annotations

"""
C-1〜C-6 一気通貫 runner

実行（リポジトリルートから）:
  python layer1_5/src/phase_C/run_all_C.py

要件:
- sys.path 調整で layer1_5 import 可能にする
- cwd を layer1_5/ にして相対パス解決
- C1→C2→C3→C4→C5→C6 の順に実行
- 最後に phase_C_summary.md を生成
- attack_angle 欠損率ログを出す（overall/two_strike×ohtani/benchmark）
"""

import os
import sys
import importlib
from pathlib import Path


MODULES = [
    "layer1_5.src.phase_C.C1_twostrike_band_penalty",
    "layer1_5.src.phase_C.C2_bandwidth_sensitivity",
    "layer1_5.src.phase_C.C3_twostrike_outband_drivers",
    "layer1_5.src.phase_C.C4_multi_kpi_band_penalty",
    "layer1_5.src.phase_C.C5_shape_by_pitch_or_zone",
    "layer1_5.src.phase_C.C6_ohtani_action_diagnosis",
    "layer1_5.src.phase_C.make_phase_C_summary",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root / "layer1_5")

    # 欠損率ログ（安全装置）
    from layer1_5.src.phase_C.utils import load_band_bounds, load_contact_quality, add_band_flags

    df = add_band_flags(load_contact_quality(), load_band_bounds())
    oht = df[df["player_id"] == "ohtani"]
    bench = df
    for gname, gdf in [("ohtani", oht), ("benchmark_group", bench)]:
        for cname, sdf in [("overall", gdf), ("two_strike", gdf[gdf["two_strike"] == True])]:  # noqa: E712
            miss = 1.0 - float(sdf["attack_angle_available"].mean()) if len(sdf) else float("nan")
            print(f"[LOG] attack_angle missing rate | {gname} | {cname}: {miss:.3f} (n={len(sdf)})")

    for m in MODULES:
        mod = importlib.import_module(m)
        print(f"\n=== RUN {m} ===")
        mod.main()

    # 生成物一覧
    out_root = repo_root / "layer1_5" / "data" / "output"
    plot_root = repo_root / "layer1_5" / "data" / "output" / "plots" / "phase_C"
    doc_path = repo_root / "layer1_5" / "docs" / "phase_C_summary.md"

    print("\n=== GENERATED FILES (phase_C) ===")
    for p in sorted(out_root.glob("C*.csv")):
        print(str(p))
    for p in sorted(plot_root.glob("C*.png")):
        print(str(p))
    print(str(doc_path))


if __name__ == "__main__":
    main()


