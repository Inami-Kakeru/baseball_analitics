## 本リポジトリの全体像（最上位ドキュメント）

この `docs/` は、**ここまでの分析（設計意図・定義・処理手順・成果物）**と、**ファイル構造（どこに何があり、何を残し、何を捨てたか）**を、第三者が追跡・再現・レビューできる形で記録するための上位ドキュメントです。

### 目的（このdocsで答えること）
- **何を分析したのか**：High-Value Zone（HVZ）をアンカーに、2ストライク条件での「帯域外（out_band）が追加的損失を生むか」を、循環論法を避けて検証する。
- **何が“定義として固定”なのか**：y₁（HVZ所属）、帯域（attack_angle band）、分母（attack_angle非欠損）など。
- **どのファイルを見れば何が分かるのか**：Layer1/Layer1.5/Phase（rev）それぞれの入口と出口。
- **データ・生成物の扱い方**：巨大CSV・rawデータをGitに入れない方針と、その理由。

### 目次
- `docs/architecture.md`
  - 研究設計（Layer構造）と“固定条件”一覧
  - 主要定義（HVZ, V, y₁, in_band/out_band, adverseの扱い）
  - 解析の因果主張の範囲（しないこと／すること）
- `docs/reproducibility.md`
  - セットアップ、推奨実行順、実行コマンド例
  - よくある詰まり（Windows/フォント/パス/欠損）
- `docs/file_structure.md`
  - ディレクトリ構造の説明（layer1 / layer1_5 / notebooks / docs）
  - 追跡対象（Git）と非追跡対象（.gitignore）の整理
- `docs/figures_rev.md`
  - `layer1_5/data/output/plots/rev/` の全図の意味と読み方
  - 「Δが大きい」解釈の分解（差・比・絶対値・公平性チェック）

### 既存の詳細ドキュメントとの関係
- Layer1（HVZ定義）の詳細は `layer1/docs/step1_high_value_zone/` にまとまっています（本docsは“上位まとめ”）。
- Layer1.5の概念・変数根拠・Layer2への橋渡しは `layer1_5/docs/` にまとまっています（本docsは“全体を通した再現導線”）。


