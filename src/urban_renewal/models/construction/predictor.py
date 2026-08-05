from __future__ import annotations

import torch


class ConstructionPredictor:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict(self, rgb, dataset: str, texture=None):
        output = self.model(rgb.to(self.device), dataset, None if texture is None else texture.to(self.device))
        return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in output.items()}
