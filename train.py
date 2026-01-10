from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.01,
) -> Tuple[List[float], List[float]]:
    """Trenuje model QTSA i zwraca historię strat (train, test)."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    train_history: List[float] = []
    test_history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0
        count_train = 0
        pbar = tqdm(train_loader, desc=f"Epoka {epoch}/{epochs}", leave=False)
        for xb, yb in pbar:
            # Ensure float32 dtype
            xb = xb.to(dtype=torch.float32)
            yb = yb.to(dtype=torch.float32)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * len(xb)
            count_train += len(xb)
            pbar.set_postfix(train_loss=loss.item())

        train_loss = total_train / max(count_train, 1)

        model.eval()
        total_test = 0.0
        count_test = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(dtype=torch.float32)
                yb = yb.to(dtype=torch.float32)
                preds = model(xb).squeeze()
                loss = loss_fn(preds, yb)
                total_test += loss.item() * len(xb)
                count_test += len(xb)
        test_loss = total_test / max(count_test, 1)

        train_history.append(train_loss)
        test_history.append(test_loss)
        print(
            f"Epoka {epoch}/{epochs} | "
            f"train MSE={train_loss:.4f} | test MSE={test_loss:.4f}"
        )

    return train_history, test_history
