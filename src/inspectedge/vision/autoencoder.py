"""Baseline B: convolutional autoencoder; anomaly map = per-pixel reconstruction error.

Requires the [ml] extra. Trained on defect-free images only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from inspectedge.vision.base import score_from_heatmaps, smooth_and_resize


def _down(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU())


class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(_down(3, 32), _down(32, 64), _down(64, 128), _down(128, 256))
        self.decoder = nn.Sequential(
            _up(256, 128),
            _up(128, 64),
            _up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderModel:
    method = "autoencoder"

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
        smoothing_sigma: float = 4.0,
    ) -> None:
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.seed, self.device, self.smoothing_sigma = seed, device, smoothing_sigma
        self.net = ConvAutoencoder().to(device)
        self.train_losses: list[float] = []

    @staticmethod
    def _to_tensor(images: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0

    def fit(self, images: np.ndarray) -> AutoencoderModel:
        g = torch.Generator().manual_seed(self.seed)
        x_all = self._to_tensor(images)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(x_all), generator=g)
            total = 0.0
            for i in range(0, len(x_all), self.batch_size):
                x = x_all[perm[i : i + self.batch_size]].to(self.device)
                loss = nn.functional.mse_loss(self.net(x), x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
            self.train_losses.append(total / len(x_all))
        return self

    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        maps = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                x = self._to_tensor(images[i : i + self.batch_size]).to(self.device)
                err = ((self.net(x) - x) ** 2).mean(dim=1).cpu().numpy()
                maps.extend(smooth_and_resize(e, e.shape, self.smoothing_sigma) for e in err)
        heatmaps = np.stack(maps).astype(np.float32)
        return score_from_heatmaps(heatmaps), heatmaps

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), d / "autoencoder.pt")
        meta = {
            "method": self.method,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "seed": self.seed,
            "smoothing_sigma": self.smoothing_sigma,
            "train_losses": self.train_losses,
        }
        (d / "ae_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> AutoencoderModel:
        d = Path(directory)
        meta = json.loads((d / "ae_meta.json").read_text())
        m = cls(
            meta["epochs"],
            meta["batch_size"],
            meta["lr"],
            meta["seed"],
            device,
            meta["smoothing_sigma"],
        )
        m.net.load_state_dict(
            torch.load(d / "autoencoder.pt", map_location=device, weights_only=True)
        )
        return m
