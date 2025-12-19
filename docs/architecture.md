## 研究設計（Layer構造）と固定条件（超重要）

本プロジェクトは、StatcastのBIP（インプレー打球）を材料に、**「高価値打球の安定性」**を操作可能変数（スイング特性）に接続するための分析パイプラインです。

### 全体の狙い（論理の骨格）
- **Layer1（HVZ定義）**：結果指標（xwOBA等）に頼らず、EV×LAの物理空間から「高価値帯（High-Value Zone）」を定義する。
- **Layer1.5（物理特性スクリーニング）**：HVZに「入る確率（y₁）」と「入った中の価値（y₂）」、および「不利条件での再現性（y₁ | adverse）」を分離して観測し、候補変数を機械的に絞る。
- **Phase（rev図＝提出用）**：2ストライク条件での **帯域外（out_band）** が、同条件内でのxSLG等に与える“追加的損失”を示し、比較・頑健性・公平性チェックまで含めて可視化する。

---

## 固定条件（分析の不変アンカー）

### 1) HVZ（High-Value Zone）の定義はLayer1で固定
- HVZはEV×LAのビン上で \(E[V]\) を推定し、高価値セル集合として確定する。
- Layer1.5以降では **HVZ自体を再定義しない**（アンカーの破壊を防ぐ）。
- 詳細は `layer1/docs/step1_high_value_zone/`。

### 2) 目的変数の役割分離（混ぜない）
- **y₁（主目的）**：HVZに入ったか（成功状態の二値、Layer1で確定）
- **y₂（補助目的）**：HVZ内の価値（\(V\) の連続値評価：天井/純度）
- **“y₃”は作らない**：不利条件は目的変数ではなく **観測窓（フィルタ）**として扱う
  - 例：\(P(y₁=1 \mid strikes=2)\)

### 3) in_band / out_band の唯一の定義（分母含む）
帯域は攻角（`attack_angle`）上の固定範囲で定義する。

- **帯域ソース（単一ソース固定）**：`layer1_5/data/output/ohtani_attack_angle_band_rate.csv`
  - `lower_bound`, `upper_bound`
  - **再推定・再学習しない**
- **分母（最重要）**：`attack_angle` **非欠損のみ**
  - 欠損を out_band 扱いしない（評価分母に入れない）
- **in_band**：
  - `attack_angle` 非欠損 かつ `lower_bound <= attack_angle <= upper_bound`
- **out_band**：
  - `attack_angle` 非欠損 かつ `attack_angle < lower_bound または attack_angle > upper_bound`

> 実装上の「唯一の正」は `layer1_5/src/phase_C/utils.py` の docstring に明記されています。

---

## 主要指標・定義

### 長打スコア \(V\)（Layer1の連続値目的）
- 定義：\(V = \max(bases - 1, 0)\)
  - out/単打 → 0
  - 二塁打 → 1
  - 三塁打 → 2
  - 本塁打 → 3

### EV×LAのビン設計（Layer1）
- bin幅：EV 2mph × LA 2deg
- 各セル：count と \(E[V]\)
- countが小さいセルは除外（ノイズ支配を避ける）

### xSLG（提出図でのKPI）
Phase（rev）では、帯域内外の「追加的損失」を示すために、`estimated_slg_using_speedangle` を xSLG として利用する。

---

## “やらないこと”（誤解とリークを避ける）
- **循環論法の回避**：
  - HVZをEV×LAで定義した後、同じEV×LA系ML指標（xwOBA等）を「HVZの説明」に持ち込んで結論にしない。
- **目的と手段の混線回避**：
  - “y₃”を別目的変数として作らない（評価条件フィルタに留める）。
- **因果主張の抑制**：
  - 本分析は観察的。施策はLayer2で提案し、ここでは「候補の絞り込み」と「頑健性の確認」を主とする。


