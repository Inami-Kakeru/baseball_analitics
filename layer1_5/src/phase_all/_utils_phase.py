from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Band:
    lower: float
    upper: float


def read_band_from_single_source(band_csv: Path) -> Band:
    """帯域の唯一の定義ソースから lower/upper を読む。"""
    b = pd.read_csv(band_csv)
    if "lower_bound" not in b.columns or "upper_bound" not in b.columns:
        raise KeyError("band csv must contain lower_bound and upper_bound")
    return Band(float(b["lower_bound"].iloc[0]), float(b["upper_bound"].iloc[0]))


def choose_primary_kpi_col(df: pd.DataFrame) -> tuple[str, str]:
    """
    Primary KPIを選択。
    - まず xSLG系（estimated_slg...）を優先
    - なければ V
    戻り値: (kpi_col_name, kpi_label)
    """
    # xSLG候補
    candidates = [
        "estimated_slg_using_speedangle",
        "estimated_slg",
        "estimated_slg_using_speedangle_",
    ]
    for c in candidates:
        if c in df.columns:
            return c, "xSLG"
    # prefix search
    for c in df.columns:
        if str(c).startswith("estimated_slg"):
            return c, "xSLG"
    if "V" in df.columns:
        return "V", "V"
    raise KeyError("No primary KPI column found (expected estimated_slg* or V)")


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.astype(float)
    return a.astype(float) / (b2.replace(0, np.nan))


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def q_edges_from_series(x: pd.Series, q: int = 10) -> np.ndarray:
    """overall分布基準の分位境界。ユニーク不足で境界が潰れることは許容。"""
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


def approx_n_ge_min(counts: pd.Series, min_n: int) -> bool:
    """「概ね」: 80%のビンがmin_n以上。"""
    if len(counts) == 0:
        return False
    return float((counts >= min_n).mean()) >= 0.8


