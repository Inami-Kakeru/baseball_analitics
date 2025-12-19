from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CONTACT_PATH = Path("data/intermediate/contact_quality.csv")
BAND_SOURCE = Path("data/output/ohtani_attack_angle_band_rate.csv")  # 単一ソース（唯一の正）


@dataclass(frozen=True)
class Band:
    lower: float
    upper: float


def load_contact_quality() -> pd.DataFrame:
    df = pd.read_csv(CONTACT_PATH)
    required = ["player_id", "strikes", "attack_angle", "y1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"contact_quality.csv 必須列が不足: {missing}")
    return df


def load_band_bounds() -> Band:
    b = pd.read_csv(BAND_SOURCE)
    if "lower_bound" not in b.columns or "upper_bound" not in b.columns:
        raise KeyError("帯域単一ソースに lower_bound / upper_bound がありません。即停止。")
    lower = float(b["lower_bound"].iloc[0])
    upper = float(b["upper_bound"].iloc[0])
    return Band(lower=lower, upper=upper)


def add_band_flags(df: pd.DataFrame, band: Band) -> pd.DataFrame:
    """
    定義（唯一の正）:
    - two_strike = (strikes == 2)
    - in_band の分母は attack_angle 非欠損のみ
    - in_band = lower<=attack_angle<=upper （attack_angle非欠損行に対して）
    - out_band = (attack_angle非欠損) かつ (in_band==False)
    """
    df = df.copy()
    df["two_strike"] = df["strikes"] == 2

    aa_ok = df["attack_angle"].notna()
    in_band = (df["attack_angle"] >= band.lower) & (df["attack_angle"] <= band.upper)
    df["in_band"] = aa_ok & in_band
    df["out_band"] = aa_ok & (~in_band)
    df["attack_angle_available"] = aa_ok
    return df


def choose_xslg_col(df: pd.DataFrame) -> str:
    # C系では xSLG 必須（無ければ即停止）
    col = "estimated_slg_using_speedangle"
    if col not in df.columns:
        raise KeyError(f"xSLG必須列 '{col}' がありません（C系は即停止）。")
    return col


def spearman_corr_no_scipy(x: pd.Series, y: pd.Series) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def q_edges(x: pd.Series, q: int = 10) -> np.ndarray:
    probs = np.linspace(0, 1, q + 1)
    edges = x.quantile(probs, interpolation="linear").to_numpy(dtype=float)
    edges = edges[~np.isnan(edges)]
    edges = np.unique(edges)
    return edges


def assign_bins_by_edges(x: pd.Series, edges: np.ndarray) -> pd.Series:
    if len(edges) < 2:
        return pd.Series([np.nan] * len(x), index=x.index)
    b = pd.cut(x.astype(float), bins=edges, include_lowest=True, right=True, labels=False)
    return b.astype("float") + 1


def approx_n_ge_min(counts: pd.Series, min_n: int = 30) -> bool:
    if len(counts) == 0:
        return False
    return float((counts >= min_n).mean()) >= 0.8


