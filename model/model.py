import torch.nn as nn


class BanknoteCNN(nn.Module):
    """
    3-block CNN for binary banknote classification (fake vs real).

    Input shape:  (batch, 3, 48, 48)
    Output shape: (batch, 2)  — logits for [fake, real]

    Architecture:
        Block 1: Conv(3→32)   + BN + ReLU + MaxPool  →  24×24
        Block 2: Conv(32→64)  + BN + ReLU + MaxPool  →  12×12
        Block 3: Conv(64→128) + BN + ReLU + AvgPool  →   2×2
        Classifier: Flatten → Linear(512→256) → Dropout → Linear(256→2)
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 48×48 → 24×24
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 24×24 → 12×12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 12×12 → 2×2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),

            # Classifier head
            nn.Flatten(),
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
