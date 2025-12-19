# Layer 1.5: 特徴量スクリーニング層

## 目的

Layer 1.5は、Layer 1（状態定義層）とLayer 2（説明モデル層）の橋渡しを担う。

- **Layer 1からの入力**: 高価値帯域ラベル `y` が付与された打球データ
- **Layer 1.5の役割**: スイング・バットトラッキング指標から、`y` を説明する候補特徴量を特定
- **Layer 2への出力**: 説明可能性の高い特徴量セット

## Layer 1との接続

- 入力: `data/input/bip_with_v_and_y.csv` (Layer 1の成果物)
- 前提: 各打球に `y` (高価値帯域到達ラベル) が付与済み

## Layer 2への橋渡し

- 出力: `data/output/candidate_features.csv`
- 各特徴量の効果量・安定性・解釈可能性を評価
- Layer 2（GA2M等）で使用する特徴量を事前選定

## 処理フロー

1. **00_prepare_layer1_5.py**: Layer 1データの前処理
2. **01_define_spin_proxies.py**: 回転の代理変数構築
3. **02_define_contact_quality.py**: 接触質指標の構築
4. **03_single_var_screening.py**: 単変量スクリーニング
5. **04_stability_check.py**: 安定性検証
6. **05_export_for_layer2.py**: Layer 2用特徴量セット確定

