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
from pathlib import Path
from queue import Queue

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context


ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path.home() / ".cache" / "sam" / "sam_vit_h_4b8939.pth"
app = Flask(__name__, static_folder=None)
LOCAL_PORT = int(os.environ.get("AETHERMAP_PORT", "8767"))
model_lock = threading.Lock()
road_lock = threading.Lock()
samroad_lock = threading.Lock()
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


def prompted_geometry(mask, feature_type, seed, bbox, width, height):
    """Convert one user-seeded SAM mask to the geometry of the chosen class."""
    import cv2
    import numpy as np

    binary = mask.astype(np.uint8)
    if feature_type == "building":
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 40 or area > binary.size * 0.15:
            return None
        epsilon = max(1.0, 0.006 * cv2.arcLength(contour, True))
        ring = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).tolist()
        if len(ring) < 3:
            return None
        ring.append(ring[0])
        return {"type": "Polygon", "coordinates": [pixel_to_lonlat(ring, bbox, width, height)]}

    paths = trace_skeleton(binary, min_pixels=12)
    if not paths:
        return None
    x, y = seed
    ranked = []
    for path in paths:
        distance = min((px - x) ** 2 + (py - y) ** 2 for px, py in path)
        length = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
        ranked.append((distance, -length, path))
    distance, _, path = min(ranked)
    if distance > 30 ** 2 or len(path) < 12:
        return None
    return {"type": "LineString", "coordinates": pixel_to_lonlat(path, bbox, width, height)}


def road_centerlines(probability, threshold=0.5, min_pixels=12):
    """Skeletonize predicted road pixels into ordered centerline paths."""
    import cv2
    import numpy as np

    binary = (probability >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= 20:
            cleaned[labels == component] = 1
    paths = trace_skeleton(cleaned, min_pixels)
    results = []
    for path in paths:
        points = cv2.approxPolyDP(np.asarray(path, dtype=np.float32).reshape(-1, 1, 2), 1.0, False).reshape(-1, 2)
        if len(points) < 2:
            continue
        score = float(np.mean([probability[y, x] for x, y in path]))
        results.append((score, [(float(x), float(y)) for x, y in points]))
    return sorted(results, key=lambda item: -len(item[1]))


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/api/health")
def health():
    return jsonify(app="aethermap-road-demo", port=LOCAL_PORT)


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


@app.get("/api/road-status")
def road_status():
    try:
        import torch  # noqa: F401
        from road_model import weights_ready
        return jsonify(available=True, checkpoint_cached=weights_ready(), model="Road segmentation TorchScript", device="CPU")
    except ImportError as exc:
        return jsonify(available=False, reason=f"Missing Python package: {exc.name}"), 503


@app.get("/api/samroad-status")
def samroad_status():
    from samroad_model import availability

    available, reason = availability()
    return jsonify(available=available, reason=reason, model="SAM-Road CityScale ViT-B", device="CUDA if available, otherwise CPU")


@app.get("/api/reference-roads")
def reference_roads():
    """Small-area Census TIGERweb roads for visual comparison and snapping."""
    import requests

    try:
        bbox = [float(v) for v in request.args["bbox"].split(",")]
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise ValueError("Invalid bounding box")
        west, south, east, north = bbox
        if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
            raise ValueError("Invalid bounding box")
        if east - west > 0.03 or north - south > 0.03:
            raise ValueError("Zoom in before loading Census reference roads")
    except (KeyError, ValueError) as exc:
        return jsonify(error=str(exc)), 400

    try:
        features = []
        base = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_PhysicalFeatures/MapServer"
        for layer_id, road_type in ((1, "primary"), (3, "secondary"), (5, "residential")):
            response = requests.get(
                f"{base}/{layer_id}/query",
                params={"f": "geojson", "where": "1=1", "geometry": ",".join(map(str, bbox)),
                        "geometryType": "esriGeometryEnvelope", "inSR": "4326", "outSR": "4326",
                        "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "true",
                        "outFields": "OBJECTID,NAME,MTFCC"},
                headers={"User-Agent": "AetherMapFeatureTracer/0.3"}, timeout=30,
            )
            response.raise_for_status()
            if len(response.content) > 12_000_000:
                raise ValueError("Census response too large; zoom in")
            data = response.json()
            if data.get("type") != "FeatureCollection":
                raise ValueError(f"Census road layer {layer_id} returned an invalid response")
            for feature in data.get("features", []):
                geometry = feature.get("geometry") or {}
                lines = [geometry.get("coordinates")] if geometry.get("type") == "LineString" else (
                    geometry.get("coordinates", []) if geometry.get("type") == "MultiLineString" else [])
                props = feature.get("properties") or {}
                for line in lines:
                    if line and len(line) >= 2:
                        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": line},
                                         "properties": {"census_id": props.get("OBJECTID") or feature.get("id"),
                                                        "highway": road_type, "name": props.get("NAME") or "",
                                                        "mtfcc": props.get("MTFCC")}})
            if len(features) > 5000:
                raise ValueError("Too many Census roads; zoom in")
        return jsonify(type="FeatureCollection", features=features, source="U.S. Census Bureau TIGERweb")
    except (requests.RequestException, ValueError) as exc:
        return jsonify(error=f"Census road reference unavailable: {exc}"), 502


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


@app.post("/api/sam-point")
def sam_point():
    """Trace only the image object selected by a user click; SAM has no class labels."""
    import numpy as np
    from PIL import Image

    origin = request.headers.get("Origin")
    if origin and origin != f"http://127.0.0.1:{LOCAL_PORT}":
        return jsonify(error="This local model only accepts requests from its own page"), 403
    if request.content_length and request.content_length > 8_000_000:
        return jsonify(error="Image upload is too large"), 413
    try:
        feature_type = request.form["feature_type"]
        if feature_type not in {"road", "building"}:
            raise ValueError("Choose road or building")
        bbox = [float(v) for v in json.loads(request.form["bbox3857"])]
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Invalid bounding box")
        Image.MAX_IMAGE_PIXELS = 2_000_000
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        if image.width != image.height or image.width < 256 or image.width > 1024:
            raise ValueError("Expected a square 256-1024 pixel image")
        x, y = float(request.form["seed_x"]), float(request.form["seed_y"])
        if not all(math.isfinite(v) for v in (x, y)) or not (0 <= x < image.width and 0 <= y < image.height):
            raise ValueError("Selected point is outside the image")
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        return jsonify(error=str(exc)), 400

    if not model_lock.acquire(blocking=False):
        return jsonify(error="Another SAM trace is running"), 409
    try:
        generator = load_generator()
        predictor = generator.predictor
        predictor.set_image(np.asarray(image))
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        choices = []
        for mask, score in zip(masks, scores):
            geometry = prompted_geometry(mask, feature_type, (x, y), bbox, image.width, image.height)
            if geometry:
                choices.append((float(score), geometry))
        if not choices:
            return jsonify(error="No usable outline at that point. Choose a clearer spot or trace manually."), 422
        score, geometry = max(choices, key=lambda item: item[0])
        return jsonify(geometry=geometry, feature_type=feature_type, score=round(score, 3), source="usgs_naip_sam_point_prompt", review_status="unverified")
    except Exception as exc:
        app.logger.exception("SAM point trace failed")
        return jsonify(error=f"SAM point trace failed: {exc}"), 500
    finally:
        model_lock.release()


@app.post("/api/road-detect")
def road_detect():
    """Infer road pixels from the submitted NAIP image and return centerlines."""
    from PIL import Image

    origin = request.headers.get("Origin")
    if origin and origin != f"http://127.0.0.1:{LOCAL_PORT}":
        return jsonify(error="This local model only accepts requests from its own page"), 403
    if request.content_length and request.content_length > 8_000_000:
        return jsonify(error="Image upload is too large"), 413
    try:
        bbox = [float(v) for v in json.loads(request.form["bbox3857"])]
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Invalid bounding box")
        threshold = float(request.form.get("threshold", 0.5))
        if not math.isfinite(threshold) or not 0.2 <= threshold <= 0.9:
            raise ValueError("Road threshold must be between 0.2 and 0.9")
        Image.MAX_IMAGE_PIXELS = 2_000_000
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        if image.width != image.height or image.width not in {512, 1024}:
            raise ValueError("Expected a 512 or 1024 pixel square image")
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        return jsonify(error=str(exc)), 400

    def generate():
        if not road_lock.acquire(blocking=False):
            yield event("error", message="Another road scan is already running")
            return
        try:
            from road_model import predict_road_probability

            yield event("status", message="Loading road-specific model…")
            probability = predict_road_probability(image)
            yield event("status", message="Extracting road centerlines from predicted road pixels…")
            paths = road_centerlines(probability, threshold)
            for score, path in paths[:200]:
                coordinates = pixel_to_lonlat(path, bbox, 256, 256)
                if len(coordinates) > 1:
                    yield event("line", coordinates=coordinates, confidence=round(score, 3))
            yield event("done", lines=min(len(paths), 200), model="road_segmentation_BAH", threshold=threshold)
        except Exception as exc:
            app.logger.exception("Road detection failed")
            yield event("error", message=f"Road detection failed: {exc}")
        finally:
            road_lock.release()

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/api/samroad-detect")
def samroad_detect():
    """Return topology-head road graph paths predicted from the submitted NAIP image."""
    from PIL import Image

    origin = request.headers.get("Origin")
    if origin and origin != f"http://127.0.0.1:{LOCAL_PORT}":
        return jsonify(error="This local model only accepts requests from its own page"), 403
    if request.content_length and request.content_length > 8_000_000:
        return jsonify(error="Image upload is too large"), 413
    try:
        bbox = [float(v) for v in json.loads(request.form["bbox3857"])]
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Invalid bounding box")
        Image.MAX_IMAGE_PIXELS = 2_000_000
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        if image.width != image.height or image.width not in {512, 1024}:
            raise ValueError("Expected a 512 or 1024 pixel square image")
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        return jsonify(error=str(exc)), 400

    def generate():
        if not samroad_lock.acquire(blocking=False):
            yield event("error", message="Another SAM-Road scan is already running")
            return
        messages = Queue()

        def run_model():
            try:
                from samroad_model import graph_paths, predict_road_graph

                messages.put(event("status", message="Loading verified SAM-Road CityScale graph model…"))
                _, edges = predict_road_graph(image, progress=lambda phase, current, total:
                    messages.put(event("progress", phase=phase, current=current, total=total)))
                messages.put(event("status", message="Connecting predicted road graph edges…"))
                paths = graph_paths(edges)
                for path, score in paths[:200]:
                    coordinates = pixel_to_lonlat(path, bbox, image.width, image.height)
                    messages.put(event("line", coordinates=coordinates, confidence=round(score, 3)))
                messages.put(event("done", lines=min(len(paths), 200), graph_edges=len(edges), model="SAM-Road CityScale ViT-B"))
            except Exception as exc:
                app.logger.exception("SAM-Road detection failed")
                messages.put(event("error", message=f"SAM-Road detection failed: {exc}"))
            finally:
                samroad_lock.release()
                messages.put(None)

        threading.Thread(target=run_model, daemon=True).start()
        while True:
            message = messages.get()
            if message is None:
                break
            yield message

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    app.run(host="127.0.0.1", port=LOCAL_PORT, threaded=True, debug=False)
