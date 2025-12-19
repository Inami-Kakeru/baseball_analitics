"""
Step 3: EV×LA価値マップから高価値帯域を定義し、各打球にラベルyを付与する（選手別対応）。
- 入力: data/processed/bip_with_v_and_y.csv, data/processed/value_map.csv
- 出力: data/processed/bip_with_v_and_y.csv（yを更新して上書き保存）
"""

from __future__ import annotations

import pandas as pd

from utils import assign_high_value_label


BIP_PATH = "data/processed/bip_with_v_and_y.csv"
VALUE_MAP_PATH = "data/processed/value_map.csv"
TOP_QUANTILE = 0.15


def assign_labels_by_player(
    bip: pd.DataFrame, value_map: pd.DataFrame, top_quantile: float = TOP_QUANTILE
) -> pd.DataFrame:
    """選手別に高価値帯域を定義してラベルを付与"""
    if "player_id" not in value_map.columns or "player_id" not in bip.columns:
        # player_idがない場合は従来通り全体で処理
        return assign_high_value_label(bip, value_map, top_quantile=top_quantile)
    
    # 選手別に処理
    bip_labeled_list = []
    for player_id in sorted(bip["player_id"].unique()):
        bip_player = bip[bip["player_id"] == player_id].copy()
        vm_player = value_map[value_map["player_id"] == player_id].copy()
        
        if len(vm_player) == 0:
            bip_player["y"] = 0
            bip_labeled_list.append(bip_player)
            continue
        
        # player_idカラムを一時的に削除してassign_high_value_labelに渡す
        vm_player_no_id = vm_player.drop(columns=["player_id"])
        bip_player_labeled = assign_high_value_label(
            bip_player, vm_player_no_id, top_quantile=top_quantile
        )
        bip_labeled_list.append(bip_player_labeled)
    
    return pd.concat(bip_labeled_list, ignore_index=True)


def main() -> None:
    bip = pd.read_csv(BIP_PATH)
    value_map = pd.read_csv(VALUE_MAP_PATH)
    bip_labeled = assign_labels_by_player(bip, value_map, top_quantile=TOP_QUANTILE)
    bip_labeled.to_csv(BIP_PATH, index=False)
    
    if "player_id" in bip_labeled.columns:
        n_players = bip_labeled["player_id"].nunique()
        print(f"updated y label ({n_players} players) -> {BIP_PATH}")
    else:
        print(f"updated y label -> {BIP_PATH}")


if __name__ == "__main__":
    main()

