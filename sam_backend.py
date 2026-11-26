"""Local SAM-assisted tracing for AetherMap. Never exposed beyond loopback.

SAM segments generic objects. Shape filtering and skeleton tracing yield
*unverified line candidates*, not road classifications.
"""

from __future__ import annotations

import io
import json
import math
import os
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context


ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path.home() / ".cache" / "sam" / "sam_vit_h_4b8939.pth"
app = Flask(__name__, static_folder=None)
LOCAL_PORT = int(os.environ.get("AETHERMAP_PORT", "8767"))
model_lock = threading.Lock()
mask_generator = None


def event(kind: str, **payload) -> str:
    return json.dumps({"type": kind, **payload}, separators=(",", ":")) + "\n"


def load_generator():
    global mask_generator
    if mask_generator is None:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"SAM checkpoint missing: {CHECKPOINT}")
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        # The local ViT-H checkpoint exceeds the GTX 1650's practical VRAM.
        model = sam_model_registry["vit_h"](checkpoint=str(CHECKPOINT))
        model.to(device="cpu")
        model.eval()
        mask_generator = SamAutomaticMaskGenerator(
            model=model,
            points_per_side=8,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.85,
            crop_n_layers=0,
            min_mask_region_area=80,
        )
    return mask_generator


def trace_skeleton(mask, min_pixels: int = 25):
    """Walk each skeleton edge once; return paths in image pixel order."""
    import numpy as np
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask.astype(bool))
    height, width = skeleton.shape
    active = set(np.flatnonzero(skeleton).tolist())
    if not active:
        return []

    def neighbors(index):
        y, x = divmod(index, width)
        return [
            ny * width + nx
            for ny in range(max(0, y - 1), min(height, y + 2))
            for nx in range(max(0, x - 1), min(width, x + 2))
            if (ny != y or nx != x) and ny * width + nx in active
        ]

    adjacency = {index: neighbors(index) for index in active}
    visited = set()
    paths = []

    def walk(start, next_index):
        path = [start, next_index]
        visited.add(frozenset((start, next_index)))
        previous, current = start, next_index
        while len(adjacency[current]) == 2:
            candidate = next(n for n in adjacency[current] if n != previous)
            edge = frozenset((current, candidate))
            if edge in visited:
                break
            visited.add(edge)
            path.append(candidate)
            previous, current = current, candidate
        if len(path) >= min_pixels:
            paths.append([(index % width, index // width) for index in path])

    for index in active:
        if len(adjacency[index]) != 2:
            for next_index in adjacency[index]:
                if frozenset((index, next_index)) not in visited:
                    walk(index, next_index)
    for index in active:
        for next_index in adjacency[index]:
            if frozenset((index, next_index)) not in visited:
                walk(index, next_index)
    return paths


def pixel_to_lonlat(path, bbox, width, height):
    left, bottom, right, top = bbox
    radius = 6378137.0
    result = []
    for x, y in path:
        mx = left + (x + 0.5) / width * (right - left)
        my = top - (y + 0.5) / height * (top - bottom)
        lon = math.degrees(mx / radius)
        lat = math.degrees(2 * math.atan(math.exp(my / radius)) - math.pi / 2)
        result.append([round(lon, 7), round(lat, 7)])
    return result


def candidate_paths(mask, min_pixels):
    """Mirror the QGIS scanner's elongated-mask heuristic, then centerline it."""
    import cv2
    import numpy as np

    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 80 or area > binary.size * 0.30:
        return []
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    if min(w, h) < 1 or max(w, h) / min(w, h) < 2.0:
        return []
    return trace_skeleton(binary, min_pixels)


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/api/sam-status")
def sam_status():
    try:
        import segment_anything  # noqa: F401
        import torch  # noqa: F401
        available = CHECKPOINT.is_file()
        reason = "ready" if available else f"Checkpoint missing: {CHECKPOINT}"
    except ImportError as exc:
        available, reason = False, f"Missing Python package: {exc.name}"
    return jsonify(available=available, loaded=mask_generator is not None, reason=reason, model="SAM ViT-H", device="CPU")


@app.get("/api/osm-roads")
def osm_roads():
    """Small-area reference geometry; this endpoint never edits OSM."""
    import requests

    try:
        bbox = [float(v) for v in request.args["bbox"].split(",")]
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise ValueError("Invalid bounding box")
        west, south, east, north = bbox
        if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
            raise ValueError("Invalid bounding box")
        if east - west > 0.03 or north - south > 0.03:
            raise ValueError("Zoom in before loading OSM reference roads")
    except (KeyError, ValueError) as exc:
        return jsonify(error=str(exc)), 400

    try:
        response = requests.get(
            "https://api.openstreetmap.org/api/0.6/map",
            params={"bbox": ",".join(map(str, bbox))},
            headers={"User-Agent": "AetherMapRoadCandidateExplorer/0.2 (local review tool)"},
            timeout=30,
        )
        response.raise_for_status()
        if len(response.content) > 12_000_000:
            raise ValueError("OSM response too large; zoom in")
        root = ET.fromstring(response.content)
        nodes = {node.get("id"): [float(node.get("lon")), float(node.get("lat"))] for node in root.findall("node")}
        features = []
        for way in root.findall("way"):
            tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
            if "highway" not in tags or tags["highway"] in {"footway", "path", "cycleway", "steps", "pedestrian", "proposed", "construction"}:
                continue
            coords = [nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes]
            if len(coords) >= 2:
                features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": {"osm_id": way.get("id"), "highway": tags["highway"], "name": tags.get("name", "")}})
        return jsonify(type="FeatureCollection", features=features, source="OpenStreetMap API 0.6", license="ODbL")
    except (requests.RequestException, ET.ParseError, ValueError) as exc:
        return jsonify(error=f"OSM reference unavailable: {exc}"), 502


@app.post("/api/sam-trace")
def sam_trace():
    import numpy as np
    from PIL import Image

    origin = request.headers.get("Origin")
    if origin and origin != f"http://127.0.0.1:{LOCAL_PORT}":
        return jsonify(error="This local model only accepts requests from its own page"), 403
    if request.content_length and request.content_length > 8_000_000:
        return jsonify(error="Image upload is too large"), 413
    try:
        Image.MAX_IMAGE_PIXELS = 2_000_000
        bbox = json.loads(request.form["bbox3857"])
        if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
            raise ValueError("Invalid bounding box")
        bbox = [float(v) for v in bbox]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Bounding box is reversed")
        min_pixels = max(10, min(100, int(request.form.get("min_pixels", 25))))
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        if image.width != image.height or image.width < 256 or image.width > 1024:
            raise ValueError("Expected a square 256-1024 pixel image")
        pixels = np.asarray(image)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        return jsonify(error=str(exc)), 400

    def generate():
        if not model_lock.acquire(blocking=False):
            yield event("error", message="Another SAM scan is already running")
            return
        try:
            yield event("status", message="Loading local SAM model on CPU…")
            generator = load_generator()
            yield event("status", message="SAM segmenting image; this may take several minutes…")
            masks = generator.generate(pixels)
            yield event("status", message=f"SAM found {len(masks)} generic masks; tracing elongated masks…")
            count = 0
            for mask in masks:
                for path in candidate_paths(mask["segmentation"], min_pixels):
                    coordinates = pixel_to_lonlat(path, bbox, image.width, image.height)
                    if len(coordinates) < 2:
                        continue
                    count += 1
                    yield event("line", coordinates=coordinates, mask_iou=round(float(mask.get("predicted_iou", 0)), 3))
            yield event("done", masks=len(masks), lines=count)
        except Exception as exc:
            app.logger.exception("SAM scan failed")
            yield event("error", message=f"SAM scan failed: {exc}")
        finally:
            model_lock.release()

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    app.run(host="127.0.0.1", port=LOCAL_PORT, threaded=True, debug=False)
