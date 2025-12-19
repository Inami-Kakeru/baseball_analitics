from __future__ import annotations

"""
Phase 一気通貫 runner

要件:
- 00→10→11→12→13→14→99 を順に実行
- 1回のコマンドで完結
- 最後に生成ファイル一覧を標準出力へ

実行例（リポジトリルートから）:
  py layer1_5/src/phase_all/run_all_phase.py
"""

import os
import sys
import importlib
from pathlib import Path


MODULES = [
    "layer1_5.src.phase_all.00_load_and_prepare",
    "layer1_5.src.phase_all.10_A0_in_band_rate_check",
    "layer1_5.src.phase_all.11_A1_out_of_band_penalty",
    "layer1_5.src.phase_all.12_A2_bat_speed_effect_in_band",
    "layer1_5.src.phase_all.13_A3_contact_efficiency_effect_in_band",
    "layer1_5.src.phase_all.14_A4_two_strike_out_of_band_drivers",
    "layer1_5.src.phase_all.99_make_phase_summary_md",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    # import解決のためにrepo rootをsys.pathへ
    sys.path.insert(0, str(repo_root))

    # 各スクリプトは layer1_5/ をcwdとして相対パスを解決する
    os.chdir(repo_root / "layer1_5")

    for m in MODULES:
        mod = importlib.import_module(m)
        print(f"\n=== RUN {m} ===")
        mod.main()

    # 生成物一覧（repo root基準で表示）
    out_root = repo_root / "layer1_5" / "data" / "output"
    docs_root = repo_root / "layer1_5" / "docs"

    print("\n=== GENERATED FILES (layer1_5/data/output) ===")
    if out_root.exists():
        for p in sorted(out_root.rglob("*")):
            if p.is_file():
                print(str(p))

    print("\n=== GENERATED FILES (layer1_5/docs) ===")
    if docs_root.exists():
        for p in sorted(docs_root.rglob("*")):
            if p.is_file() and p.name == "phase_summary.md":
                print(str(p))


if __name__ == "__main__":
    main()


