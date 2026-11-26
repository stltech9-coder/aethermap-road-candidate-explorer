"""Road-only image segmentation for the local AetherMap service.

Model: manish2607/road_extrcation_model_BAH, MIT license. The checkpoint is
cached outside the repository and verified against its published SHA-256.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


WEIGHTS = Path.home() / ".cache" / "aethermap" / "road_model_BAH.pt"
WEIGHTS_URL = "https://huggingface.co/manish2607/road_extrcation_model_BAH/resolve/main/road_model_Deployment.pt"
WEIGHTS_SHA256 = "586384e78014471a5ca10d8e7fdd686a4cd998bd3701cd68d2ca4dc6789293d5"
WEIGHTS_SIZE = 56_805_303
_model = None


def weights_ready() -> bool:
    return WEIGHTS.is_file() and WEIGHTS.stat().st_size == WEIGHTS_SIZE


def verified_weights() -> Path:
    import requests

    if weights_ready() and hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == WEIGHTS_SHA256:
        return WEIGHTS
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    pending = WEIGHTS.with_suffix(".part")
    digest = hashlib.sha256()
    with requests.get(WEIGHTS_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with pending.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    digest.update(chunk)
                    output.write(chunk)
    if digest.hexdigest() != WEIGHTS_SHA256:
        pending.unlink(missing_ok=True)
        raise ValueError("Downloaded road model checksum mismatch")
    os.replace(pending, WEIGHTS)
    return WEIGHTS


def predict_road_probability(image):
    """Return 256×256 road probabilities for a square RGB PIL image."""
    global _model
    import numpy as np
    import torch
    from PIL import Image

    if _model is None:
        _model = torch.jit.load(str(verified_weights()), map_location="cpu")
        _model.eval()
    resized = image.resize((256, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)
    with torch.inference_mode():
        logits = _model(tensor)[0, 0]
        probability = torch.sigmoid(logits).numpy()
    return probability
