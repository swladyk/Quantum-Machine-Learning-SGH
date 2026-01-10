# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv (3.13.11)
#     language: python
#     name: python3
# ---

# %%
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from data_processor import DataProcessor
from q_model import QTSAModel
from train import train_model

# %%
# Przykładowy skrót z notatnika:
processor = DataProcessor()
processor.download_data("BTC-USD", "2022-01-01", "2025-01-01")
processor.preprocess_data(window_size=12, use_log_returns=False)
X_train, y_train, X_test, y_test = processor.get_tensors()

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train), batch_size=16, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test), batch_size=16, shuffle=False
)

model = QTSAModel(window_size=12)
train_loss, test_loss = train_model(model, train_loader, test_loader, epochs=5, lr=0.01)