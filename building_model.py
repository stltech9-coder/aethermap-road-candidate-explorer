"""Local adapter for the user's ArcGIS USA building-footprint Mask R-CNN package."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

PACKAGE = Path.home() / "Documents" / "ArcGIS" / "Deep Learning Models" / "usa_building_footprints.dlpk"
WEIGHTS = "usa_building_footprints.pth"
_model = None


def availability():
    if not PACKAGE.is_file():
        return False, f"Building checkpoint missing: {PACKAGE}"
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        return False, f"Missing Python package: {exc.name}"
    return True, "ready"


def _modern_key(key):
    key = re.sub(r"^(backbone\.fpn\.(?:inner_blocks|layer_blocks)\.\d+)\.(weight|bias)$", r"\1.0.\2", key)
    key = re.sub(r"^roi_heads\.mask_head\.mask_fcn([1-4])\.(weight|bias)$",
                 lambda match: f"roi_heads.mask_head.{int(match[1]) - 1}.0.{match[2]}", key)
    return key.replace("rpn.head.conv.weight", "rpn.head.conv.0.0.weight").replace(
        "rpn.head.conv.bias", "rpn.head.conv.0.0.bias")


def load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    if not PACKAGE.is_file():
        raise FileNotFoundError(PACKAGE)
    with zipfile.ZipFile(PACKAGE) as archive:
        weights = torch.load(io.BytesIO(archive.read(WEIGHTS)), map_location="cpu", weights_only=True)
    weights = {_modern_key(key): value for key, value in weights.items()}
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(not key.endswith("num_batches_tracked") for key in missing):
        raise ValueError(f"Building checkpoint is incompatible: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.eval()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    _model = model
    return model


def tile_starts(length, tile=512, stride=448):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def detect_buildings(pixels, bbox, score_threshold=0.65):
    """Yield unreviewed footprint polygons from 512px tiles of a square RGB image."""
    import cv2
    import numpy as np
    import torch

    from sam_backend import pixel_to_lonlat

    model = load_model()
    height, width = pixels.shape[:2]
    xs, ys = tile_starts(width), tile_starts(height)
    total = len(xs) * len(ys)
    candidates = []
    for tile_index, (top, left) in enumerate(((y, x) for y in ys for x in xs), start=1):
        chip = pixels[top:top + 512, left:left + 512]
        tensor = torch.from_numpy(np.ascontiguousarray(chip.transpose(2, 0, 1))).float() / 255.0
        with torch.inference_mode():
            prediction = model([tensor])[0]
        yield {"type": "progress", "current": tile_index, "total": total}
        for mask, score in zip(prediction["masks"], prediction["scores"]):
            confidence = float(score)
            if confidence < score_threshold:
                continue
            binary = (mask[0].numpy() >= 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < 50:
                continue
            rx, ry, rw, rh = cv2.boundingRect(contour)
            box = (left + rx, top + ry, left + rx + rw, top + ry + rh)
            epsilon = max(1.0, 0.006 * cv2.arcLength(contour, True))
            ring = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).tolist()
            if len(ring) < 3:
                continue
            ring = [[x + left, y + top] for x, y in ring]
            ring.append(ring[0])
            area = cv2.contourArea(np.asarray(ring, dtype=np.int32))
            if area < 50:
                continue
            clipped = (rx <= 1 and left > 0) or (ry <= 1 and top > 0) or (
                rx + rw >= chip.shape[1] - 1 and left + chip.shape[1] < width) or (
                ry + rh >= chip.shape[0] - 1 and top + chip.shape[0] < height)
            candidates.append({
                "type": "building",
                "geometry": {"type": "Polygon", "coordinates": [pixel_to_lonlat(ring, bbox, width, height)]},
                "confidence": round(confidence, 3),
                "tile": tile_index,
                "tiles": total,
                "pixel_ring": ring,
                "box": box,
                "area": area,
                "clipped": clipped,
            })

    for candidate in deduplicate_buildings(candidates):
        yield {key: value for key, value in candidate.items() if key not in {"box", "area", "clipped"}}


def deduplicate_buildings(candidates):
    """Keep one outline per overlapping roof, preferring intact chip detections."""
    ordered = sorted(candidates, key=lambda item: (not item["clipped"], item["confidence"], item["area"]), reverse=True)
    kept = []
    for candidate in ordered:
        if any(_same_roof(candidate, previous) for previous in kept):
            continue
        kept.append(candidate)
    return kept


def _same_roof(a, b):
    import cv2
    import numpy as np

    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return False
    size = (y2 - y1, x2 - x1)
    masks = []
    for item in (a, b):
        mask = np.zeros(size, dtype=np.uint8)
        ring = np.asarray(item["pixel_ring"], dtype=np.int32) - [x1, y1]
        cv2.fillPoly(mask, [ring], 1)
        masks.append(mask)
    overlap = int(np.count_nonzero(masks[0] & masks[1]))
    if not overlap:
        return False
    smaller = min(a["area"], b["area"])
    union = a["area"] + b["area"] - overlap
    return overlap / smaller >= 0.55 or overlap / union >= 0.35
