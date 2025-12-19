"""
Layer1.5 Bias Validation (単変量・層別・HVZ定義感度)

目的:
- Layer1.5で得た「HVZ内で効いた変数」が、HVZ選択バイアスではなく
  全BIP/不利条件/層別でも y1（成功状態）に対して頑健に残るかを検証する。

固定:
- 目的変数は y1（HVZ所属）で固定
- y3は作らない（is_adverse=True は観測窓）
- 多変量/GA2M/回帰は禁止（単変量・確率ベースのみ）

検証対象変数（固定）:
- T1系: attack_angle, swing_length, bat_speed（分母リーク注意）
- T2系: intercept_ball_minus_batter_pos_y_inches, intercept_ball_minus_batter_pos_x_inches

感度分析:
- y1(strict): HVZ上位10%
- y1(loose):  HVZ上位20%
  ※ EV×LAビン設計はLayer1のまま変更しない

出力:
- layer1_5/data/output/bias_validation_results.csv
- layer1_5/data/output/plots/bias_validation/*.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_T1 = ["attack_angle", "swing_length", "bat_speed"]
FEATURES_T2 = [
    "intercept_ball_minus_batter_pos_y_inches",
    "intercept_ball_minus_batter_pos_x_inches",
]
FEATURES = FEATURES_T1 + FEATURES_T2

STRATA_COLS = ["pitch_group", "plate_z_group", "count_group"]
ADVERSE_COL = "is_adverse"

MIN_BIN_N = 30
MIN_EFFECT_D = 0.2
MIN_RHO_ABS = 0.2  # U字/逆U字除外（弱すぎる単調性はNG）


@dataclass(frozen=True)
class EvalOut:
    bins: int
    n_total: int
    rho: float
    direction: int  # +1 / -1 / 0
    effect_d: float
    pass_flag: bool


def repo_root() -> Path:
    # .../layer1_5/src/07_bias_validation.py -> repo root
    return Path(__file__).resolve().parents[2]


def spearman_corr_no_scipy(x: pd.Series, y: pd.Series) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def choose_bins(x: pd.Series) -> int | None:
    n_unique = x.dropna().nunique()
    if n_unique >= 10:
        return 10
    if n_unique >= 5:
        return 5
    return None


def qbin(x: pd.Series, q: int) -> pd.Series:
    return pd.qcut(x, q=q, labels=False, duplicates="drop") + 1


def approx_bin_n_ok(bin_counts: pd.Series, min_n: int) -> bool:
    if len(bin_counts) == 0:
        return False
    return float((bin_counts >= min_n).mean()) >= 0.8


def cohens_d_binary(y: pd.Series, mask_hi: pd.Series, mask_lo: pd.Series) -> float:
    """
    yが0/1のとき、上位binと下位binの平均差を pooled std で標準化。
    """
    y_hi = y[mask_hi]
    y_lo = y[mask_lo]
    if len(y_hi) == 0 or len(y_lo) == 0:
        return float("nan")
    diff = float(y_hi.mean() - y_lo.mean())
    std = float(y.std(ddof=0))
    if std <= 0 or np.isnan(std):
        return float("nan")
    return diff / std


def eval_univariate_prob(df: pd.DataFrame, x_col: str, y_col: str) -> EvalOut:
    d = df[[x_col, y_col]].dropna().copy()
    n_total = int(len(d))
    if n_total < MIN_BIN_N * 3:
        return EvalOut(0, n_total, float("nan"), 0, float("nan"), False)

    q = choose_bins(d[x_col])
    if q is None:
        return EvalOut(0, n_total, float("nan"), 0, float("nan"), False)

    try:
        d["bin"] = qbin(d[x_col], q)
    except Exception:
        return EvalOut(0, n_total, float("nan"), 0, float("nan"), False)

    agg = d.groupby("bin")[y_col].agg(["mean", "count"]).reset_index().sort_values("bin")
    bins_used = int(agg["bin"].nunique())
    if bins_used < 3:
        return EvalOut(bins_used, n_total, float("nan"), 0, float("nan"), False)

    n_ok = approx_bin_n_ok(agg["count"], MIN_BIN_N)
    rho = spearman_corr_no_scipy(agg["bin"], agg["mean"])
    direction = int(np.sign(rho)) if not np.isnan(rho) else 0

    # effect_d: top bin vs bottom bin
    lo_bin = int(agg["bin"].min())
    hi_bin = int(agg["bin"].max())
    effect_d = cohens_d_binary(
        d[y_col].astype(float),
        mask_hi=(d["bin"] == hi_bin),
        mask_lo=(d["bin"] == lo_bin),
    )

    pass_flag = (
        n_ok
        and (not np.isnan(effect_d))
        and (abs(effect_d) >= MIN_EFFECT_D)
        and (not np.isnan(rho))
        and (abs(rho) >= MIN_RHO_ABS)
    )
    return EvalOut(bins_used, n_total, float(rho) if not np.isnan(rho) else float("nan"), direction, float(effect_d), bool(pass_flag))


def build_hvz_y_labels(
    contact_df: pd.DataFrame,
    value_map: pd.DataFrame,
    top_quantile: float,
    out_col: str,
) -> pd.DataFrame:
    """
    Layer1のvalue_map（player_id, ev_bin, la_bin, E_V）から、
    選手別に上位q%セルを高価値帯として y を再ラベル。
    """
    df = contact_df.copy()

    needed = {"player_id", "ev_bin", "la_bin"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"contact_quality.csvに必要列が不足: {sorted(missing)}")

    vm = value_map.copy()
    required_vm = {"player_id", "ev_bin", "la_bin", "E_V"}
    missing_vm = required_vm - set(vm.columns)
    if missing_vm:
        raise KeyError(f"value_map.csvに必要列が不足: {sorted(missing_vm)}")

    # NaNセルを除外
    vm = vm.dropna(subset=["E_V"])

    high_sets: dict[str, set[tuple[float, float]]] = {}
    for pid, g in vm.groupby("player_id"):
        if len(g) == 0:
            high_sets[pid] = set()
            continue
        thr = float(g["E_V"].quantile(1 - top_quantile))
        high = g[g["E_V"] >= thr][["ev_bin", "la_bin"]]
        high_sets[pid] = set(map(tuple, high.to_numpy()))

    def row_is_high(r: pd.Series) -> int:
        s = high_sets.get(r["player_id"], set())
        return int((r["ev_bin"], r["la_bin"]) in s)

    df[out_col] = df.apply(row_is_high, axis=1).astype(int)
    return df


def strata_consistency(
    df: pd.DataFrame, x: str, y_col: str, overall_dir: int
) -> int:
    """
    pitch/plate_z/count の各グループについて、
    レベルごとの方向の多数決がoverall_dirと一致するかを0-3でカウント。
    """
    if overall_dir == 0:
        return 0

    score = 0
    for col in STRATA_COLS:
        if col not in df.columns:
            continue
        dirs = []
        for v in df[col].dropna().unique():
            seg = df[df[col] == v]
            out = eval_univariate_prob(seg, x, y_col)
            if out.pass_flag and out.direction != 0:
                dirs.append(out.direction)
        if len(dirs) == 0:
            continue
        maj = int(np.sign(np.sum(dirs)))
        if maj != 0 and maj == overall_dir:
            score += 1
    return score


def label_from_checks(
    overall_ok: bool,
    adverse_ok: bool,
    strata_score: int,
    hvz10: bool,
    hvz20: bool,
) -> str:
    # PASS: 全体で明確 + adverse符号維持 + HVZ10/20両方で残存
    if overall_ok and adverse_ok and (strata_score >= 2) and hvz10 and hvz20:
        return "PASS"
    # CONDITIONAL: 全体弱いが adverse または層別で一貫
    if (adverse_ok and strata_score >= 2) or (overall_ok and strata_score >= 2) or (overall_ok and adverse_ok):
        return "CONDITIONAL"
    return "FAIL"


def plot_decile_curve(
    df: pd.DataFrame,
    x: str,
    y_col: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    def curve(sub: pd.DataFrame):
        d = sub[[x, y_col]].dropna().copy()
        q = choose_bins(d[x])
        if q is None:
            return None
        try:
            d["bin"] = qbin(d[x], q)
        except Exception:
            return None
        agg = d.groupby("bin")[y_col].agg(["mean", "count"]).reset_index().sort_values("bin")
        return agg

    overall = curve(df)
    adverse = curve(df[df[ADVERSE_COL] == True]) if ADVERSE_COL in df.columns else None  # noqa: E712

    plt.figure(figsize=(7, 4))
    if overall is not None:
        plt.plot(overall["bin"], overall["mean"], marker="o", label="overall")
    if adverse is not None:
        plt.plot(adverse["bin"], adverse["mean"], marker="o", label="adverse")
    plt.title(title)
    plt.xlabel("bin (quantile)")
    plt.ylabel(f"P({y_col}=1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    root = repo_root()
    contact_path = root / "layer1_5" / "data" / "intermediate" / "contact_quality.csv"
    value_map_path = root / "layer1" / "data" / "processed" / "value_map.csv"

    out_csv = root / "layer1_5" / "data" / "output" / "bias_validation_results.csv"
    plot_dir = root / "layer1_5" / "data" / "output" / "plots" / "bias_validation"

    df = pd.read_csv(contact_path)
    vm = pd.read_csv(value_map_path)

    # base y1は既存列を使用（固定）
    if "y1" not in df.columns:
        raise KeyError("contact_quality.csv に 'y1' がありません（Layer1.5の前処理を確認）。")
    if ADVERSE_COL not in df.columns:
        raise KeyError("contact_quality.csv に 'is_adverse' がありません。")

    # 感度分析用のy定義を作成（y3は作らない）
    df = build_hvz_y_labels(df, vm, top_quantile=0.10, out_col="y1_hvz10")
    df = build_hvz_y_labels(df, vm, top_quantile=0.20, out_col="y1_hvz20")

    rows = []
    for feat in FEATURES:
        if feat not in df.columns:
            raise KeyError(f"検証対象featureが見つかりません: {feat}")

        # base y1
        over = eval_univariate_prob(df, feat, "y1")
        adv = eval_univariate_prob(df[df[ADVERSE_COL] == True], feat, "y1")  # noqa: E712
        overall_dir = "+" if over.direction > 0 else "-" if over.direction < 0 else "0"
        adverse_dir = "+" if adv.direction > 0 else "-" if adv.direction < 0 else "0"
        adverse_ok = bool(adv.pass_flag and (adv.direction == over.direction) and (over.direction != 0))
        strata_score = strata_consistency(df, feat, "y1", overall_dir=over.direction)

        # hvz strict/loose PASS判定（Step A-Cを再実行して「残存」判定）
        def hvz_pass(ycol: str) -> bool:
            o = eval_univariate_prob(df, feat, ycol)
            a = eval_univariate_prob(df[df[ADVERSE_COL] == True], feat, ycol)  # noqa: E712
            if not (o.pass_flag and a.pass_flag):
                return False
            if o.direction == 0 or a.direction == 0 or o.direction != a.direction:
                return False
            sc = strata_consistency(df, feat, ycol, overall_dir=o.direction)
            return sc >= 2

        hvz10 = hvz_pass("y1_hvz10")
        hvz20 = hvz_pass("y1_hvz20")

        final_label = label_from_checks(
            overall_ok=bool(over.pass_flag),
            adverse_ok=adverse_ok,
            strata_score=strata_score,
            hvz10=hvz10,
            hvz20=hvz20,
        )

        rows.append(
            {
                "feature": feat,
                "overall_direction": overall_dir,
                "adverse_direction": adverse_dir,
                "strata_consistency": int(strata_score),
                "hvz_10pct_pass": bool(hvz10),
                "hvz_20pct_pass": bool(hvz20),
                "final_label": final_label,
                # 参考（検証ログ）
                "overall_bins": over.bins,
                "overall_effect_d": over.effect_d,
                "overall_spearman_rho": over.rho,
                "adverse_bins": adv.bins,
                "adverse_effect_d": adv.effect_d,
                "adverse_spearman_rho": adv.rho,
            }
        )

        # plots: base + hvz10 + hvz20（overall+adverse）
        plot_decile_curve(
            df,
            x=feat,
            y_col="y1",
            title=f"{feat} | y1 (base)",
            out_path=plot_dir / f"{feat}__y1_base.png",
        )
        plot_decile_curve(
            df,
            x=feat,
            y_col="y1_hvz10",
            title=f"{feat} | y1 (HVZ top 10%)",
            out_path=plot_dir / f"{feat}__y1_hvz10.png",
        )
        plot_decile_curve(
            df,
            x=feat,
            y_col="y1_hvz20",
            title=f"{feat} | y1 (HVZ top 20%)",
            out_path=plot_dir / f"{feat}__y1_hvz20.png",
        )

    out = pd.DataFrame(rows).sort_values(["final_label", "feature"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}")
    print(out[["feature", "final_label", "overall_direction", "adverse_direction", "strata_consistency", "hvz_10pct_pass", "hvz_20pct_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()


