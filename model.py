from __future__ import annotations

import torch
import pennylane as qml


class QTSA(torch.nn.Module):
    """Jednobitowy model QTSA z sekwencyjnym data re-uploading."""

    def __init__(self, window_size: int, n_layers: int = 2, device_name: str = "default.qubit"):
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size musi być dodatni.")
        if n_layers < 1:
            raise ValueError("n_layers musi być dodatni.")
        self.window_size = window_size
        self.n_layers = n_layers
        self.dev = qml.device(device_name, wires=1)

        weight_shapes = {"weights": (n_layers, window_size)}
        self.qnode = self._build_qnode()
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            for layer in range(self.n_layers):
                for t in range(self.window_size):
                    qml.RY(inputs[t], wires=0)
                    qml.RZ(weights[layer, t], wires=0)
            return qml.expval(qml.PauliZ(0))

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        z_expectation = self.qlayer(x)
        return (z_expectation + 1.0) * 0.5 * torch.pi
