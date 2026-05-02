# neural.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Dataset for one-step prediction: u^n -> u^{n+1}
class HeatEquationDataset(Dataset):
    """
    Dataset mapping:
        u^n -> u^{n+1}
    """

    def __init__(self, u_hist: np.ndarray):
        """
        Input:
            u_hist: array of shape (nt+1, nx)
        """
        X = u_hist[:-1]
        Y = u_hist[1:]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_dataloader(u_hist, batch_size=32, shuffle=True):
    dataset = HeatEquationDataset(u_hist)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Neural network architecture
class TimeStepperNN(nn.Module):
    """
    Learns:
        u^{n+1} = NN(u^n)
    """

    def __init__(self, nx: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(nx, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, nx),
        )

    def forward(self, x):
        return self.net(x)


# Training loop
def train_model(model: nn.Module, dataloader: DataLoader, epochs: int = 50, lr: float = 1e-3):
    """
    Train for one-step prediction.

    Returns:
        trained model
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in dataloader:
            pred = model(x)

            # enforce boundary conditions
            pred[:, 0] = 0.0
            pred[:, -1] = 0.0

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6e}")

    return model