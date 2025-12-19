## 採用した in_band_rate 定義（唯一の正）

**in_band_rate は「attack_angle が記録されている打球のみ」を分母とし、帯域は `ohtani_attack_angle_band_rate.csv` の `lower_bound/upper_bound`（単一ソース）で固定した上で、\(\mathrm{lower}\le attack\_angle\le \mathrm{upper}\) を満たす割合として定義する。**

---

## 再計算後の in_band_rate テーブル（採用定義）

出力CSV: `layer1_5/data/output/in_band_rate_unified.csv`

---

## スライド用説明文（1〜2文）

- **in_band_rate の分母は「attack_angle が記録されている打球」に限定し、attack_angle 欠損は分母から除外（out_band扱いしない）しています。**  
- **帯域は `ohtani_attack_angle_band_rate.csv` の lower/upper を単一ソースとして固定し、再学習は行っていません。**

---

## 捨てるべき in_band_rate 定義（採用しない理由）

- **~0.60 系の値を“別定義のin_band_rate”として併記・参考掲載しません。**  
  理由は、それらが **分母（欠損の扱い）や適用対象が暗黙に異なる**形で算出され、スライド上で「定義の揺れ」と誤解されやすく、本研究の主張に耐えないためです。


