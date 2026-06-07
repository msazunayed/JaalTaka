from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import BanknoteCNN

SAVED_DIR = Path(__file__).parent / "saved"
WEIGHTS   = SAVED_DIR / "best_model.pth"
META_FILE = SAVED_DIR / "model_meta.json"


def build_transform(size: int) -> transforms.Compose:
    """Standard image preprocessing: resize, to tensor, normalize to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class BanknotePredictor:
    """Loads the trained model once and exposes a predict() method."""

    def __init__(self):
        with open(META_FILE) as f:
            self.meta = json.load(f)

        self.class_names = self.meta["class_names"]  # ['fake', 'real']
        self.input_size  = self.meta.get("input_size", 48)
        self.transform   = build_transform(self.input_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = BanknoteCNN(num_classes=self.meta["num_classes"])
        self.model.load_state_dict(
            torch.load(WEIGHTS, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()

    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on a PIL image.

        Returns:
            {
                "label":         "fake" or "real",
                "confidence":    float (0–1),
                "probabilities": {"fake": float, "real": float}
            }
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)[0]

        predicted_idx   = probs.argmax().item()
        predicted_label = self.class_names[predicted_idx]

        return {
            "label":         predicted_label,
            "confidence":    round(probs[predicted_idx].item(), 4),
            "probabilities": {
                cls: round(probs[i].item(), 4)
                for i, cls in enumerate(self.class_names)
            },
        }


# Module-level singleton — loaded once, reused across requests
_predictor: BanknotePredictor | None = None


def get_predictor() -> BanknotePredictor:
    """Return the shared predictor instance, creating it on first call."""
    global _predictor
    if _predictor is None:
        _predictor = BanknotePredictor()
    return _predictor


def predict_image(image: Image.Image) -> Dict:
    """Convenience wrapper used by the FastAPI route."""
    return get_predictor().predict(image)
