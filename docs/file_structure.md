## ファイル構造（どこに何があるか）

このプロジェクトは大きく `layer1/` と `layer1_5/` に分かれます。

---

## ルート直下
- `PROJECT_OVERVIEW.md`
  - プロジェクト概要（高レベル）
- `notebooks/`
  - Layer1（EV×LA可視化など）の探索用ノートブック
- `docs/`
  - **本書**（全体の設計・再現導線・成果物の読み方）

---

## `layer1/`（HVZ定義＝アンカー）

### 役割
- EV×LA 空間における価値マップ（\(E[V]\)）を作り、HVZセル集合と y（=後段のy₁）を付与する。

### 主な場所
- `layer1/src/`
  - `00_load_and_clean.py`：Statcast rawの読込・整形（複数選手対応）
  - `01_make_long_hit_score.py`：長打スコア \(V\) を付与
  - `02_ev_la_value_map.py`：EV×LAビンで \(E[V]\) を推定（選手別も可）
  - `03_define_high_value_zone.py`：HVZ定義＆ラベル付与（y）
  - `utils.py`：共通関数
- `layer1/docs/step1_high_value_zone/`
  - Step1（HVZ定義）の詳細ドキュメント（設計思想・定義・制約）
- `layer1/data/`
  - `raw/`：Statcastの元データ（例：`statcast_ohtani.csv`）
  - `processed/`：Layer1成果物（大きいのでGit管理しない運用を推奨）

---

## `layer1_5/`（物理特性スクリーニング＋2スト帯域分析）

### 役割
- Layer1成果物（BIPにVとyが付いている）を受け取り、y₁/y₂/観測窓（is_adverse）を固定。
- 回転“そのもの”ではなく代理変数（spin proxy）を構築し、単変量で機械的に候補を絞る。
- 2スト条件での **帯域内/外（in_band/out_band）** を固定定義し、提出用（rev）図を生成する。

### 主な場所
- `layer1_5/src/`
  - `00_prepare_layer1_5.py`：前処理（y₁,y₂,is_adverse,層別ラベル）
  - `01_define_spin_proxies.py`：回転代理変数の生成
  - `02_define_contact_quality.py`：接触質派生指標
  - `03_single_var_screening.py`：分位×確率/期待値の単変量スクリーニング
  - `04_stability_check.py`：安定性分類（A/B/C/Dなど）
  - `05_export_for_layer2.py`：Layer2へ渡す候補変数の確定
  - `06_univariate_screening_targets.py`：T1/T2（機構理解用）の単変量スクリーニング
  - `10_ohtani_attack_angle_band_rate.py`：固定帯域（attack_angle band）を出力（decile固定）
  - `phase_all/`：一括実行用の集計・図生成（基盤）
  - `phase_C/`：提出図（rev含む）生成とユーティリティ
- `layer1_5/docs/`
  - Layer1.5の概念・変数根拠・レビュー用プロンプト・まとめ
- `layer1_5/data/`
  - `input/`：Layer1成果物の受け渡し（運用ではGit管理しない）
  - `intermediate/`：中間生成物（運用ではGit管理しない）
  - `output/`
    - `ohtani_attack_angle_band_rate.csv`：**帯域定義の単一ソース**
    - `plots/rev/`：**提出用（rev）図一式（これが“本質”の成果物）**

---

## Git管理方針（重要）

### なぜデータをGitに入れないのか
- Statcast rawはサイズが大きく、配布・再取得も可能で、履歴管理の対象ではない。
- 中間生成物は再生成できる一方で、巨大化しやすくレビューのノイズになる。
- よって **コード・ドキュメント・最終図（rev）** を主に追跡し、データは別管理（ローカル）に寄せる。

### 具体（.gitignore）
- `layer1/data/raw/` は運用上はローカル置き（必要なら別途配布）
- `layer1_5/data/input/` `layer1_5/data/intermediate/` は追跡しない
- `layer1_5/data/output/` は原則追跡しないが、**rev図**と**帯域CSV**等は例外として残す


