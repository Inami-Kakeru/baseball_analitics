"""
Layer1.5: Univariate screening for mechanism targets (T1/T2)

目的:
- y1（成功状態）は固定。y3は作らず is_adverse=True を観測窓として扱う。
- 中間ターゲットとして
  T1 = contact_efficiency = launch_speed / bat_speed
  T2 = carry_efficiency   = hit_distance_sc / launch_speed
  を用い、操作可能変数XがT1/T2に与える単変量の単調効果を機械的に絞り込む。

入力:
- data/intermediate/contact_quality.csv

出力:
- data/output/univariate_targets_screening.csv
  columns:
    feature,target,segment,bins,effect_d,spearman_rho,direction,n_total,pass_flag
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


IN_PATH = os.path.join("data", "intermediate", "contact_quality.csv")
OUT_PATH = os.path.join("data", "output", "univariate_targets_screening.csv")

# 固定ターゲット（中間）
T1_COL = "contact_efficiency"
T2_COL = "carry_efficiency"

# 操作可能説明変数（固定）
BASE_FEATURES = ["attack_angle", "swing_path_tilt", "swing_length", "bat_speed"]
INTERCEPT_PREFIX = "intercept_ball_"

# セグメント
SEGMENT_COLS = ["pitch_group", "plate_z_group", "count_group"]
ADVERSE_FLAG_COL = "is_adverse"

# 採用基準（機械的）
MIN_BIN_N = 30
MIN_EFFECT_D = 0.2
MIN_RHO_ABS = 0.2  # Spearmanの符号が意味を持つ最低ライン（U字の排除にも寄与）


@dataclass(frozen=True)
class ScreeningResult:
    feature: str
    target: str
    segment: str
    bins: int
    effect_d: float
    spearman_rho: float
    direction: str  # "+" or "-"
    n_total: int
    pass_flag: bool


def spearman_corr_no_scipy(x: pd.Series, y: pd.Series) -> float:
    """scipy不要のSpearman（rank→pearson）。"""
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def monotonicity_check(bin_means: np.ndarray) -> tuple[float, int]:
    """
    ビン平均の一次差分から
    - dir_strength: 増加/減少の優勢度
    - sign_changes: 符号反転回数（U字検出）
    """
    diffs = np.diff(bin_means.astype(float))
    if len(diffs) == 0:
        return 0.0, 99

    pos = float(np.mean(diffs > 0))
    neg = float(np.mean(diffs < 0))
    dir_strength = max(pos, neg)

    signs = np.sign(diffs)
    signs = signs[signs != 0]
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0) if len(signs) > 1 else 0)
    return dir_strength, sign_changes


def is_clear_u_shape(bin_means: np.ndarray, rho: float) -> bool:
    """
    明確なU字・逆U字の除外:
    - Spearmanが弱い（|rho| < MIN_RHO_ABS）なら非単調（U字含む）として扱う
    """
    if np.isnan(rho):
        return True
    return abs(rho) < MIN_RHO_ABS


def choose_bins(x: pd.Series) -> int | None:
    """
    原則10分位。ユニーク不足なら5分位へ。
    """
    n_unique = x.dropna().nunique()
    if n_unique >= 10:
        return 10
    if n_unique >= 5:
        return 5
    return None


def bin_series(x: pd.Series, q: int) -> pd.Series:
    """q分位ビン（ユニーク不足時のdrop対応）。"""
    # labels 1..k（dropによりk<qになる場合あり）
    return pd.qcut(x, q=q, labels=False, duplicates="drop") + 1


def approx_bin_n_ok(bin_counts: pd.Series, min_n: int) -> bool:
    """
    「概ねn>=30」: 80%以上のビンがmin_n以上ならOK
    """
    if len(bin_counts) == 0:
        return False
    return float((bin_counts >= min_n).mean()) >= 0.8


def compute_effect_d(bin_means: pd.Series, target_std: float) -> float:
    if np.isnan(target_std) or target_std <= 0:
        return float("nan")
    return float((bin_means.max() - bin_means.min()) / target_std)


def evaluate_feature_target(
    df: pd.DataFrame, feature: str, target: str, segment_name: str
) -> ScreeningResult:
    d = df[[feature, target]].dropna().copy()
    n_total = int(len(d))
    if n_total < MIN_BIN_N * 3:
        return ScreeningResult(
            feature=feature,
            target=target,
            segment=segment_name,
            bins=0,
            effect_d=float("nan"),
            spearman_rho=float("nan"),
            direction="?",
            n_total=n_total,
            pass_flag=False,
        )

    q = choose_bins(d[feature])
    if q is None:
        return ScreeningResult(
            feature=feature,
            target=target,
            segment=segment_name,
            bins=0,
            effect_d=float("nan"),
            spearman_rho=float("nan"),
            direction="?",
            n_total=n_total,
            pass_flag=False,
        )

    try:
        d["bin"] = bin_series(d[feature], q)
    except Exception:
        return ScreeningResult(
            feature=feature,
            target=target,
            segment=segment_name,
            bins=0,
            effect_d=float("nan"),
            spearman_rho=float("nan"),
            direction="?",
            n_total=n_total,
            pass_flag=False,
        )

    # bin集計
    agg = d.groupby("bin")[target].agg(["mean", "count"]).reset_index()
    agg = agg.sort_values("bin")
    bin_means = agg["mean"]
    bin_counts = agg["count"]

    # n条件
    n_ok = approx_bin_n_ok(bin_counts, MIN_BIN_N)

    # 効果量（セグメント内std）
    target_std = float(d[target].std(ddof=0))
    effect_d = compute_effect_d(bin_means, target_std)

    # 単調性
    rho = spearman_corr_no_scipy(agg["bin"], agg["mean"])
    dir_strength, sign_changes = monotonicity_check(bin_means.to_numpy())
    direction = "+" if rho > 0 else "-" if rho < 0 else "?"

    # pass判定（機械的・ローカル）
    # - 各binで概ね n>=30（80%ルール）
    # - |d|>=0.2
    # - Spearmanの符号が意味を持つ（|rho|>=0.2）＝明確なU字/逆U字を除外
    pass_flag = (
        n_ok
        and (not np.isnan(effect_d))
        and (abs(effect_d) >= MIN_EFFECT_D)
        and (not is_clear_u_shape(bin_means.to_numpy(dtype=float), rho))
    )

    return ScreeningResult(
        feature=feature,
        target=target,
        segment=segment_name,
        bins=int(agg["bin"].nunique()),
        effect_d=float(effect_d) if not np.isnan(effect_d) else float("nan"),
        spearman_rho=float(rho) if not np.isnan(rho) else float("nan"),
        direction=direction,
        n_total=n_total,
        pass_flag=bool(pass_flag),
    )


def iter_segments(df: pd.DataFrame):
    # overall
    yield "overall", df
    # adverse window
    if ADVERSE_FLAG_COL in df.columns:
        yield "is_adverse=True", df[df[ADVERSE_FLAG_COL] == True]  # noqa: E712
    # strata columns
    for col in SEGMENT_COLS:
        if col in df.columns:
            for v in sorted(df[col].dropna().unique()):
                yield f"{col}={v}", df[df[col] == v]


def controllable_features(df: pd.DataFrame) -> list[str]:
    feats = []
    for f in BASE_FEATURES:
        if f in df.columns:
            feats.append(f)
    feats.extend([c for c in df.columns if c.startswith(INTERCEPT_PREFIX)])
    # 重複排除 + 安定ソート
    feats = sorted(set(feats))
    return feats


def main() -> None:
    df = pd.read_csv(IN_PATH)

    required = [
        T1_COL,
        T2_COL,
        "attack_angle",
        "swing_path_tilt",
        "swing_length",
        "bat_speed",
        "pitch_group",
        "plate_z_group",
        "count_group",
        ADVERSE_FLAG_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    feats = controllable_features(df)
    targets = [("T1", T1_COL), ("T2", T2_COL)]

    results: list[ScreeningResult] = []
    for seg_name, seg_df in iter_segments(df):
        for feat in feats:
            for tgt_name, tgt_col in targets:
                r = evaluate_feature_target(seg_df, feat, tgt_col, seg_name)
                results.append(
                    ScreeningResult(
                        feature=r.feature,
                        target=tgt_name,
                        segment=r.segment,
                        bins=r.bins,
                        effect_d=r.effect_d,
                        spearman_rho=r.spearman_rho,
                        direction=r.direction,
                        n_total=r.n_total,
                        pass_flag=r.pass_flag,
                    )
                )

    out = pd.DataFrame([r.__dict__ for r in results])
    out.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()


