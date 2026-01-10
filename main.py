import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_processor import DataProcessor
from q_model import QTSAModel
from train import train_model
from visualize import plot_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="QTSA (Serial Data Re-uploading, 1 kubit) z PennyLane + PyTorch."
    )
    parser.add_argument("--ticker", default="BTC-USD", help="Ticker z Yahoo Finance.")
    parser.add_argument("--start-date", default="2022-01-01", help="Data startowa (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2025-01-01", help="Data końcowa (YYYY-MM-DD).")
    parser.add_argument("--window-size", type=int, default=20, help="Długość okna czasowego L.")
    parser.add_argument("--epochs", type=int, default=30, help="Liczba epok treningu.")
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar batcha.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--log-returns",
        action="store_true",
        help="Użyj log-returns zamiast cen zamknięcia (stabilniejsze, trudniejsze do interpretacji cenowo).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    processor = DataProcessor()
    processor.download_data(args.ticker, args.start_date, args.end_date)
    # Domyślnie używamy cen zamknięcia, aby móc łatwo odwrócić skalowanie do cen.
    processor.preprocess_data(window_size=args.window_size, use_log_returns=args.log_returns)
    X_train, y_train, X_test, y_test = processor.get_tensors()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False
    )

    model = QTSAModel(window_size=args.window_size)
    train_loss, test_loss = train_model(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Predykcja na zbiorze testowym (w skali [0, pi]).
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(dtype=torch.float32)
            preds_scaled.append(model(xb).squeeze())
    preds_scaled = torch.cat(preds_scaled).cpu().numpy().reshape(-1, 1)

    y_train_scaled = y_train.cpu().numpy().reshape(-1, 1)
    y_test_scaled = y_test.cpu().numpy().reshape(-1, 1)

    # Odwrócenie skalowania do realnych cen.
    scaler = processor.scaler
    if scaler is None:
        raise ValueError("Brak dopasowanego skalera.")

    y_true_full_scaled = np.concatenate([y_train_scaled, y_test_scaled], axis=0)
    y_true_full = scaler.inverse_transform(y_true_full_scaled).flatten()
    y_pred_full = np.concatenate(
        [
            np.full((len(y_train_scaled),), np.nan, dtype=float),
            scaler.inverse_transform(preds_scaled).flatten(),
        ],
        axis=0,
    )

    split_idx = len(y_train_scaled)
    plot_results(y_true_full, y_pred_full, train_loss, test_loss, split_idx)


if __name__ == "__main__":
    main()
