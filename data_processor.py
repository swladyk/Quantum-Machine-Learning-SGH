from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    """Pobieranie, skalowanie i budowa sekwencji dla QTSA."""

    def __init__(self):
        self.raw_series: Optional[pd.Series] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.window_size: Optional[int] = None
        self.use_log_returns: bool = False
        self._scaled_values: Optional[np.ndarray] = None
        self._X_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._X_test: Optional[torch.Tensor] = None
        self._y_test: Optional[torch.Tensor] = None

    def download_data(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        """Pobiera ceny zamknięcia z yfinance."""
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty or "Close" not in df:
            raise ValueError(f"Brak danych Close dla tickera {ticker}.")
        self.raw_series = df["Close"].dropna()
        return self.raw_series

    def _compute_series(self) -> pd.Series:
        if self.raw_series is None:
            raise ValueError("Najpierw wywołaj download_data.")
        if self.use_log_returns:
            returns = np.log(self.raw_series / self.raw_series.shift(1)).dropna()
            return returns
        return self.raw_series

    def preprocess_data(self, window_size: int, use_log_returns: bool = False):
        """Skalowanie do [0, pi] i budowa sekwencji X(t-window...t-1) -> y(t)."""
        if window_size < 1:
            raise ValueError("window_size musi być dodatni.")

        self.window_size = window_size
        self.use_log_returns = use_log_returns

        series = self._compute_series()
        values = series.values.reshape(-1, 1)

        self.scaler = MinMaxScaler(feature_range=(0, math.pi))
        scaled = self.scaler.fit_transform(values).astype(np.float32)
        self._scaled_values = scaled.squeeze()

        if len(self._scaled_values) <= window_size:
            raise ValueError("Za krótki szereg czasowy względem window_size.")

        X_seq, y_seq = [], []
        for idx in range(window_size, len(self._scaled_values)):
            X_seq.append(self._scaled_values[idx - window_size : idx])
            y_seq.append(self._scaled_values[idx])

        X_seq = np.stack(X_seq)
        y_seq = np.array(y_seq, dtype=np.float32)

        split_idx = int(len(X_seq) * 0.8)
        self._X_train = torch.tensor(X_seq[:split_idx], dtype=torch.float32)
        self._y_train = torch.tensor(y_seq[:split_idx], dtype=torch.float32)
        self._X_test = torch.tensor(X_seq[split_idx:], dtype=torch.float32)
        self._y_test = torch.tensor(y_seq[split_idx:], dtype=torch.float32)

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zwraca (X_train, y_train, X_test, y_test) jako tensory PyTorch."""
        for attr in (self._X_train, self._y_train, self._X_test, self._y_test):
            if attr is None:
                raise ValueError("Najpierw wywołaj preprocess_data.")
        return self._X_train, self._y_train, self._X_test, self._y_test
