## 1. 目的（1行）
two_strike（strikes==2）で、attack_angleの固定帯域を外さない制御と、帯域内での出力を両輪として長打力向上を意思決定に落とす。

## 2. 主張（1本）
**長打を増やす鍵は「attack_angleの固定帯域を外さない（制御）」×「帯域内でEV/出力を上げる（出力）」であり、特にtwo_strikeでの再現性を最重要ターゲットとする。**

## 3. 事実①：帯域内率（A0）
| group           | condition   |     n |   in_band_rate |   lower_bound |   upper_bound |
|:----------------|:------------|------:|---------------:|--------------:|--------------:|
| ohtani          | overall     |  2034 |          0.285 |         7.485 |        19.751 |
| ohtani          | two_strike  |   767 |          0.274 |         7.485 |        19.751 |
| benchmark_group | overall     | 11057 |          0.277 |         7.485 |        19.751 |
| benchmark_group | two_strike  |  4255 |          0.268 |         7.485 |        19.751 |

## 4. 事実②：帯域外ペナルティ（A1）
Primary KPI: **xSLG**（使用列: `estimated_slg_using_speedangle`）
| group           | condition   |   n_total |   n_in |   n_out |   mean_primary_kpi_in |   mean_primary_kpi_out |   diff_primary_kpi |   ratio_primary_kpi |   py1_in |   py1_out |   diff_py1 |
|:----------------|:------------|----------:|-------:|--------:|----------------------:|-----------------------:|-------------------:|--------------------:|---------:|----------:|-----------:|
| ohtani          | overall     |      2033 |    579 |    1454 |                 1.014 |                  0.806 |              0.208 |               1.258 |    0.226 |     0.145 |      0.081 |
| ohtani          | two_strike  |       766 |    210 |     556 |                 0.963 |                  0.71  |              0.252 |               1.355 |    0.224 |     0.119 |      0.105 |
| benchmark_group | overall     |     11054 |   3067 |    7987 |                 0.874 |                  0.735 |              0.139 |               1.189 |    0.196 |     0.139 |      0.058 |
| benchmark_group | two_strike  |      4254 |   1142 |    3112 |                 0.813 |                  0.672 |              0.14  |               1.208 |    0.185 |     0.117 |      0.067 |

## 5. 事実③：帯域内の出力レバー（A2）
`in_band==True` のみで、bat_speedを分位化してKPIとP(y1=1)を確認。
出力CSV: `data/output/A2_bat_speed_effect_in_band.csv` / 図: `data/output/plots/A2_bat_speed_effect_in_band.png`

## 6. 事実④：帯域内の効率特徴（A3）
`in_band==True` のみで、contact_efficiency（launch_speed/bat_speed）を分位化してKPIとP(y1=1)を確認。
注意：これは相関の観察であり、因果は主張しない。
出力CSV: `data/output/A3_contact_efficiency_effect_in_band.csv` / 図: `data/output/plots/A3_contact_efficiency_effect_in_band.png`

## 7. 事実⑤：two_strike帯域外ドライバー（A4）
two_strikeのみに限定し、out_band（帯域外）になりやすい特徴を分位×確率で確認。
出力CSV: `data/output/A4_two_strike_out_of_band_drivers.csv` / 図: `data/output/plots/A4_two_strike_out_of_band_drivers.png`

## 8. 提案（現場アクション：仮説ベースの介入）
- **制御（攻角帯域）**: 固定帯域からの逸脱を減らす練習設計（two_strikeを優先窓にする）。
- **出力（bat_speed）**: 帯域内に入った打球での出力上振れを狙う（帯域外に出ないことを前提条件にする）。
- **インパクト条件（効率）**: contact_efficiencyが高い局面の特徴を、フォーム・タイミング・当て方の観察指標として使う（因果断定はしない）。
- **two_strike特化ドリル**: A4で候補になった特徴（ブレの兆候）を減らす練習を優先（原因の断定はしない）。

## 9. リスク/限界
- データは上位打者に偏ったbenchmark_groupであり、一般化しない。
- 単変量・分位×確率の観察であり、因果や最適値は主張しない。

## 10. 次フェーズ
- 球種・コース別の最適帯域は“補助”として検討（本筋に混ぜない）。
