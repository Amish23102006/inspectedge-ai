"""ImageNet-pretrained torchvision feature extractor (requires the [ml] extra).

Note: the first run downloads backbone weights from download.pytorch.org.
"""

from __future__ import annotations

import numpy as np


class TorchvisionFeatureExtractor:
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] | list[str] = ("layer2", "layer3"),
        device: str = "cpu",
    ) -> None:
        import torch
        import torchvision
        from torchvision.models.feature_extraction import create_feature_extractor

        weights = torchvision.models.get_model_weights(backbone).DEFAULT
        model = torchvision.models.get_model(backbone, weights=weights).eval().to(device)
        self._fx = create_feature_extractor(model, return_nodes={name: name for name in layers})
        self._torch = torch
        self.layers = list(layers)
        self.device = device
        self.weights_name = str(weights)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def describe(self) -> dict:
        return {"backbone_weights": self.weights_name, "layers": self.layers}

    def __call__(self, images: np.ndarray) -> np.ndarray:
        torch = self._torch
        f = torch.nn.functional
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0
            x = (x.to(self.device) - self._mean) / self._std
            out = self._fx(x)
            feats = [f.avg_pool2d(out[name], 3, stride=1, padding=1) for name in self.layers]
            target = feats[0].shape[-2:]
            feats = [
                t
                if t.shape[-2:] == target
                else f.interpolate(t, size=target, mode="bilinear", align_corners=False)
                for t in feats
            ]
            return torch.cat(feats, dim=1).cpu().numpy().astype(np.float32)
