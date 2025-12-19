"""
安定性検証: 条件を変えても効果が残るかの安定性検証。
- 入力: data/output/screening_results.csv
- 出力: data/output/feature_classification.csv（A/B/C/D分類）
"""

from __future__ import annotations

import pandas as pd
import numpy as np


SCREENING_PATH = "data/output/screening_results.csv"
OUT_PATH = "data/output/feature_classification.csv"
THRESHOLDS_PATH = "config/thresholds.yaml"


def _load_thresholds() -> tuple[float, float]:
    # (min_effect_range, min_stability)
    try:
        import yaml

        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        min_effect = float(cfg["thresholds"]["screening"]["min_effect_size"])
        min_stability = float(cfg["thresholds"]["screening"]["stability_threshold"])
        return min_effect, min_stability
    except Exception:
        return 0.1, 0.8


def _direction(df: pd.DataFrame) -> float:
    # decile と metric の相関（符号を方向として扱う）
    if df["decile"].nunique() < 2:
        return 0.0
    return float(np.sign(df["decile"].corr(df["metric"])))


def classify_features(screening: pd.DataFrame) -> pd.DataFrame:
    """
    A/B/C/D分類（簡易版）:
    - A: y1 と y1|adverse が両方 strong & stable
    - B: y2 のみ strong & stable
    - C: overall弱いが、特定segmentで強い（条件付き）
    - D: それ以外
    """
    min_effect, min_stability = _load_thresholds()

    required_cols = {"segment", "target", "feature", "decile", "metric", "n"}
    missing = required_cols - set(screening.columns)
    if missing:
        raise KeyError(f"screening_results.csv に必要列が不足: {sorted(missing)}")

    # overall の効果（rangeと方向）
    overall = screening[screening["segment"] == "overall"].copy()
    if overall.empty:
        raise ValueError("overallセグメントがありません。03_single_var_screening.py の出力を確認してください。")

    def overall_stats(g: pd.DataFrame) -> pd.Series:
        eff_range = g["metric"].max() - g["metric"].min()
        return pd.Series(
            {
                "overall_range": float(eff_range),
                "overall_dir": _direction(g),
            }
        )

    overall_summary = (
        overall.groupby(["target", "feature"], as_index=False)
        .apply(lambda x: overall_stats(x))
        .reset_index(drop=True)
    )

    # segmentごとの方向一致率（stability）
    def stability_for(tf: tuple[str, str]) -> float:
        target, feature = tf
        g_all = screening[(screening["target"] == target) & (screening["feature"] == feature)]
        g_over = g_all[g_all["segment"] == "overall"]
        if g_over.empty:
            return 0.0
        over_dir = _direction(g_over)
        if over_dir == 0:
            return 0.0

        dirs = []
        for seg, g_seg in g_all.groupby("segment"):
            if seg == "overall":
                continue
            if g_seg["n"].sum() < 30:
                continue
            d = _direction(g_seg)
            if d != 0:
                dirs.append(d)
        if not dirs:
            return 0.0
        return float((np.array(dirs) == over_dir).mean())

    overall_summary["stability"] = overall_summary.apply(
        lambda r: stability_for((r["target"], r["feature"])), axis=1
    )
    overall_summary["strong"] = (overall_summary["overall_range"] >= min_effect) & (
        overall_summary["stability"] >= min_stability
    )

    # feature単位でカテゴリ判定
    feat_rows = []
    for feature in sorted(overall_summary["feature"].unique()):
        s_y1 = overall_summary[(overall_summary["feature"] == feature) & (overall_summary["target"] == "y1")]
        s_y1a = overall_summary[(overall_summary["feature"] == feature) & (overall_summary["target"] == "y1|adverse")]
        s_y2 = overall_summary[(overall_summary["feature"] == feature) & (overall_summary["target"] == "y2=E[V|y1=1]")]

        y1_strong = bool(len(s_y1) and s_y1["strong"].iloc[0])
        y1a_strong = bool(len(s_y1a) and s_y1a["strong"].iloc[0])
        y2_strong = bool(len(s_y2) and s_y2["strong"].iloc[0])

        # 条件付き（C）の判定: overallは弱いが、どこかのsegmentでrangeがmin_effect以上
        cond_flag = False
        for tgt in ("y1", "y1|adverse", "y2=E[V|y1=1]"):
            g = screening[(screening["target"] == tgt) & (screening["feature"] == feature)]
            if g.empty:
                continue
            # segment別のrange
            seg_ranges = g.groupby("segment")["metric"].agg(lambda x: x.max() - x.min())
            if (seg_ranges.drop(labels=["overall"], errors="ignore") >= min_effect).any():
                # overallが弱いときだけC候補
                ov = overall_summary[(overall_summary["target"] == tgt) & (overall_summary["feature"] == feature)]
                if len(ov) and not bool(ov["strong"].iloc[0]):
                    cond_flag = True
                    break

        if y1_strong and y1a_strong:
            category = "A"
        elif y2_strong and (not y1_strong) and (not y1a_strong):
            category = "B"
        elif cond_flag:
            category = "C"
        else:
            category = "D"

        feat_rows.append(
            {
                "feature": feature,
                "category": category,
                "y1_strong": y1_strong,
                "y1_adverse_strong": y1a_strong,
                "y2_strong": y2_strong,
            }
        )

    return pd.DataFrame(feat_rows)


def main() -> None:
    screening = pd.read_csv(SCREENING_PATH)
    classified = classify_features(screening)
    classified.to_csv(OUT_PATH, index=False)
    print(f"saved feature classification -> {OUT_PATH} ({len(classified)} features)")


if __name__ == "__main__":
    main()

