## 再現手順（Windows想定）

このドキュメントは「新しいPCにクローンした人」が、最小の手順で **rev図まで再現**できるようにするための実行ガイドです。

---

## 0) 前提
- Python 3.10+（推奨）
- Windowsの場合、matplotlibの日本語フォントで警告が出ることがある
  - 本プロジェクトはフォント自動選択（例：Yu Gothic）で警告を抑制する実装を含む

---

## 1) 推奨パッケージ
最低限、以下が必要です（環境によってはすでに入っています）。
- numpy
- pandas
- matplotlib
- pyyaml（configを扱う場合）

（注意）本リポジトリは “MLで当てに行く” ことを目的にしていないため、scikit-learnは必須ではありません。

---

## 2) データ配置（重要）
Gitにデータは含まれません（運用方針）。以下をローカルに置きます。

### Layer1 raw
- `layer1/data/raw/`
  - `statcast_*.csv`（選手別）

### Layer1成果物（Layer1.5の入力）
- `layer1/data/processed/`（例）
  - `bip_with_v_and_y.csv`
- これを `layer1_5/data/input/` に配置（コピーでも可）

---

## 3) Layer1 実行（HVZ定義）
リポジトリルートで実行（例）：

```bash
python layer1/src/00_load_and_clean.py
python layer1/src/01_make_long_hit_score.py
python layer1/src/02_ev_la_value_map.py
python layer1/src/03_define_high_value_zone.py
```

出力（例）：
- `layer1/data/processed/bip_with_v_and_y.csv`

※ Layer1の設計と定義は `layer1/docs/step1_high_value_zone/` を参照。

---

## 4) Layer1.5 実行（スクリーニング）
`layer1_5/` 配下へ移動して実行するのが最も安全です（相対パス前提があるため）。

```bash
cd layer1_5
python src/00_prepare_layer1_5.py
python src/01_define_spin_proxies.py
python src/02_define_contact_quality.py
python src/03_single_var_screening.py
python src/04_stability_check.py
python src/05_export_for_layer2.py
```

出力（例）：
- `layer1_5/data/output/screening_results.csv`
- `layer1_5/data/output/candidate_features.csv`

---

## 5) 固定帯域の生成（attack_angle band）
固定帯域は `decile 4〜9` を中央帯域として復元する方式で決めています：

```bash
cd layer1_5
python src/10_ohtani_attack_angle_band_rate.py
```

出力：
- `layer1_5/data/output/ohtani_attack_angle_band_rate.csv`

---

## 6) 提出用（rev）図の生成
```bash
cd layer1_5
python src/phase_C/make_rev_submission_figures.py
```

出力：
- `layer1_5/data/output/plots/rev/` 配下のPNG一式

---

## よくある詰まり

### A) `ModuleNotFoundError: layer1_5`
ルートから直実行すると相対importで失敗する場合があります。`cd layer1_5` してから実行してください。

### B) フォント警告
Windowsで `Yu Gothic` 等が見つからない場合、matplotlibが警告を出します。
本プロジェクトの `final_viz_utils.py` / `rev_viz_utils` 系はフォントを自動選択します。

### C) 2ストでサンプルが足りない
分位化や帯域内外比較は `n>=30` を目安にフィルタしています。
対象の期間・選手が少ないと、図やCSVが空になることがあります。


