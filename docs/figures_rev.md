## rev図（提出用）一覧と読み方

rev図はすべて `layer1_5/data/output/plots/rev/` に出力されます。  
共通の注記（右下）には、**条件・分母・帯域（単一ソース固定）・n** が必ず含まれます。

---

## 共通前提（全図共通で守っている定義）

### 帯域（attack_angle band）
- `layer1_5/data/output/ohtani_attack_angle_band_rate.csv` の `lower_bound/upper_bound` を使用（単一ソース）
- 以後は **再学習しない**

### 分母（重要）
- `attack_angle` **非欠損のみ**
- 欠損を out_band 扱いにしない

---

## 図の目的別ガイド

### 1) 帯域が「恣意的に作られていない」ことの確認
- `rev_fig06_attack_angle_hist_band.png`
  - 上：overall、下：two_strike
  - **灰色帯＝固定帯域**を重ね、分布の中で帯域がどこにあるかを確認する

---

### 2) 2スト内で out_band が“追加的損失”と関連するか（核）
- `rev_fig08_inout_mean_xslg_twostrike.png`
  - 大谷 vs 上位打者サンプル（2スト）
  - **ΔxSLG（内−外）** を明示し、2ストの“落ちた世界”の中でも帯域外が損失と関連することを示す

---

### 3) 選手比較：2ストでの in/out の位置（絶対値）＋差（Δ）
- `rev_fig08_twostrike_inout_xslg_player_compare.png`
  - 左：ダンベル（in/out の平均xSLG）
  - 右：Δ（内−外）
  - ここで重要なのは「Δが大きい」=「弱点」と即断しないこと。
    - **inが高いだけ**でもΔは大きくなるため、橙点（outの絶対値）を見る。

---

### 4) Δの分解：『inが強すぎる』vs『outが弱い』を判定する（比R/相対損失L）
- `rev_fig08_twostrike_relative_loss_player_compare.png`
  - **R = out / in**（追随度）
  - **L = (in − out) / in**（in基準の相対損失率）
  - 読み方：
    - Rが低い（=outがinに追随できない）→ outが弱い寄り
    - Rが高い（=outもinに近い）→ Δが大きくても「inが強いだけ」寄り

---

### 5) 公平性チェック：共通帯域（大谷固定）が“大谷に有利な定規”になっていないか
- `rev_appx_band_fairness_twostrike.png`
  - 左：共通帯域（大谷固定）
  - 右：選手別最適帯域（幅は共通帯域と同じに固定し、2ストで in平均xSLGが最大の窓を探索）
  - 目的：
    - 「共通帯域でのΔの順位」が、選手別最適帯域にするとどれだけ変わるかを見る
    - 定義バイアス（帯域が特定選手に当たりすぎ）のリスクを明示する

---

### 6) 連続的劣化：帯域中心からの距離で xSLG が連続的に落ちるか
- `rev_fig09_distance_vs_xslg_twostrike.png`
  - 横軸：\(|attack\_angle - center|\)
  - ローリング平均で「距離が大きいほど落ちやすい」傾向を示す

---

### 7) out_band の左右非対称（低側/高側）の補助確認
- `rev_fig09_out_low_high_asymmetry.png`
  - out_band のどちら側（低側/高側）がより損失に寄与するかの補助資料

---

## CSV（図の裏取り）
- `layer1_5/data/output/rev_twostrike_band_metrics_by_player.csv`
  - 共通帯域（大谷固定）での in/out/Δ/R/L の集計
- `layer1_5/data/output/rev_twostrike_band_fairness_by_player.csv`
  - 選手別最適帯域（幅固定）でのΔ等、および共通帯域Δとの比較


