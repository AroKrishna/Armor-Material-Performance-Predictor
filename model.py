"""
model.py

Neural network architecture for predicting V50 (ballistic limit velocity)
from material, projectile, and impact-scenario features.

The input size is driven by data_utils.FEATURES (currently 13: 8 material
properties + 4 projectile properties + angle) rather than being hardcoded,
so this file automatically stays in sync with data_utils.py.
"""

import torch
import torch.nn as nn


class ArmorNet(nn.Module):
    """
    Feed-forward MLP regressor for V50 prediction.

    Compared to a plain Linear-ReLU stack, this version adds:
      - BatchNorm after each hidden layer, which stabilizes and speeds up
        training on standardized tabular features.
      - Dropout for regularization, since the dataset is relatively small
        (~1000 rows) and a plain MLP of this size can overfit it.
      - Configurable hidden layer sizes / dropout rate, so architecture
        choices can be swept during experimentation without editing this
        file.
      - Explicit weight initialization for reproducibility (paired with a
        fixed random seed in train.py).

    Parameters
    ----------
    input_size : int
        Number of input features. Should be len(data_utils.FEATURES).
    hidden_sizes : tuple[int, ...]
        Sizes of the hidden layers, in order.
    dropout : float
        Dropout probability applied after each hidden activation.
        Set to 0.0 to disable dropout entirely.
    """

    def __init__(self, input_size: int, hidden_sizes=(128, 64), dropout: float = 0.15):
        super(ArmorNet, self).__init__()

        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout_rate = dropout

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        # Final regression head: single output (predicted, scaled V50)
        layers.append(nn.Linear(in_features, 1))

        self.layer = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for ReLU layers, zero bias -- deterministic
        given a fixed torch random seed set by the caller (see train.py)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def config(self) -> dict:
        """Returns the architecture config, useful for saving alongside
        model weights so the same architecture can be reconstructed later."""
        return {
            'input_size': self.input_size,
            'hidden_sizes': list(self.hidden_sizes),
            'dropout': self.dropout_rate,
        }


# ----------------------------------------------------------------------
# Self-test when run directly: python model.py
# ----------------------------------------------------------------------

if __name__ == '__main__':
    try:
        from data_utils import FEATURES
        n_features = len(FEATURES)
    except ImportError:
        print("[model] data_utils.py not found alongside this file; "
              "falling back to a placeholder input size of 13 for the self-test.")
        n_features = 13

    torch.manual_seed(42)
    model = ArmorNet(input_size=n_features)
    print(model)
    print("\nConfig:", model.config())

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    # Forward pass sanity check with a dummy batch (batch size > 1 so
    # BatchNorm1d doesn't error in train mode)
    dummy_batch = torch.randn(8, n_features)
    model.train()
    out_train = model(dummy_batch)
    print(f"\nTrain-mode output shape: {tuple(out_train.shape)}")

    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_batch)
    print(f"Eval-mode output shape:  {tuple(out_eval.shape)}")