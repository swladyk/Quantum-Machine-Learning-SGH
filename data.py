from __future__ import annotations

import math
from typing import Tuple

import pandas as pd
import torch
import yfinance as yf

# Small epsilon to prevent division by zero during scaling.
EPS = 1e-9


def fetch_close_prices(
    ticker: str, period: str = "1y", interval: str = "1d"
) -> pd.Series:
    """Download adjusted close prices for a ticker using yfinance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty or "Close" not in df:
        raise ValueError(f"Brak danych dla tickera {ticker}.")
    series = df["Close"].dropna()
    if series.empty:
        raise ValueError(f"Pusty szereg czasowy dla tickera {ticker}.")
    return series


def scale_to_pi(series: pd.Series) -> Tuple[torch.Tensor, float, float]:
    """Min-max scale values to [0, pi]. Returns scaled tensor, min, and span."""
    min_v = float(series.min())
    max_v = float(series.max())
    span = max(max_v - min_v, EPS)
    scaled = (torch.tensor(series.values, dtype=torch.float32) - min_v) / span
    return scaled * math.pi, min_v, span


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Sliding-window dataset for next-step prediction."""

    def __init__(self, values: torch.Tensor, window_size: int):
        if values.ndim != 1:
            raise ValueError("values musi być tensorem 1D.")
        if window_size < 1:
            raise ValueError("window_size musi być dodatni.")
        if len(values) <= window_size:
            raise ValueError("Za mało próbek na zadany window_size.")
        self.values = values
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.values) - self.window_size

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.window_size
        features = self.values[start:end]
        target = self.values[end]
        return features, target
