from __future__ import annotations

import torch
import pennylane as qml


class QTSAModel(torch.nn.Module):
    """QTSA z serial data re-uploading na jednym kubicie."""

    def __init__(self, window_size: int, device_name: str = "default.qubit"):
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size musi być dodatni.")
        self.window_size = window_size
        self.dev = qml.device(device_name, wires=1)

        weight_shapes = {"weights": (window_size + 1, 3)}
        self.qnode = self._build_qnode()
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def _build_qnode(self):
        window_size = self.window_size  # Capture for closure
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qtsa_circuit(inputs, weights):
            # Use captured window_size, not inputs.shape
            for i in range(window_size):
                qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=0)
                qml.RX(inputs[..., i], wires=0)
            qml.Rot(weights[-1, 0], weights[-1, 1], weights[-1, 2], wires=0)
            return qml.expval(qml.PauliZ(0))

        return qtsa_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mapuje wynik z [-1, 1] na [0, pi] by dopasować skalowanie danych."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # Ensure float32 for PennyLane compatibility
        x = x.to(dtype=torch.float32)
        z = self.qlayer(x)
        return (z + 1.0) * 0.5 * torch.pi
