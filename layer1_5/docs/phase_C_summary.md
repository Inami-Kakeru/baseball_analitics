## 目的
two_strike（strikes==2）で落ちるのは前提として認めた上で、その“中で”帯域外（out_band）が追加の損失を生むかを確認し、意思決定に使える介入対象へ接続する。

## 主要な答え（短く）
- two_strike“内”でも、in_band vs out_band で xSLG と P(y1=1) の差が残る（C1）。
- 帯域幅を±2/±4/±6度で揺らしても、差が大きく崩れないかを確認できる（C2）。

## C1: two_strike band penalty（主張の芯）
| group           |   n_attack_angle_available |   n_in |   n_out |   diff_xslg |   ratio_xslg |   diff_py1 |
|:----------------|---------------------------:|-------:|--------:|------------:|-------------:|-----------:|
| ohtani          |                        355 |    210 |     145 |       0.346 |        1.561 |      0.148 |
| benchmark_group |                       1963 |   1142 |     821 |       0.202 |        1.332 |      0.09  |

## C2: bandwidth sensitivity（恣意性の排除）
| band    | group           |   diff_xslg |
|:--------|:----------------|------------:|
| base    | ohtani          |       0.346 |
| base    | benchmark_group |       0.202 |
| narrow2 | ohtani          |       0.364 |
| narrow2 | benchmark_group |       0.161 |
| wide2   | ohtani          |       0.301 |
| wide2   | benchmark_group |       0.204 |
| narrow4 | ohtani          |       0.223 |
| narrow4 | benchmark_group |       0.115 |
| wide4   | ohtani          |       0.347 |
| wide4   | benchmark_group |       0.29  |
| narrow6 | ohtani          |      -0.262 |
| narrow6 | benchmark_group |       0.082 |
| wide6   | ohtani          |       0.378 |
| wide6   | benchmark_group |       0.331 |

## 提案への接続（因果ではなく、仮説ベースの介入）
- 介入対象は『2ストで out_band を減らす（崩れ方の低減）』。
- 成功指標（例）: two_strikeでの `diff_xslg` の縮小、`P(out_band=1)` の低下。
- benchmark_group は上位打者サンプルであり、一般化しない。
- 単変量観察の結果であり、因果は主張しない。

## 補助（幅の納得: KPI複数）
出力CSV: `data/output/C4_multi_kpi_band_penalty.csv`（図は作らない）

## 大谷のみ診断（改善プラン直結の材料）
| condition   |   n_attack_angle_available |   in_band_rate |   diff_xslg |   diff_py1 |
|:------------|---------------------------:|---------------:|------------:|-----------:|
| overall     |                        950 |          0.609 |       0.295 |      0.118 |
| two_strike  |                        355 |          0.592 |       0.346 |      0.148 |
