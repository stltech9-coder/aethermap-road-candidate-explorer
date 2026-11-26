"""Local inference adapter for the original htcr/sam_road CityScale checkpoint.

The upstream source and large checkpoints live outside Git; no SAM-Road++
checkpoint is available from its authors. This adapter runs the original
SAM-Road ViT-B mask and topology heads on NAIP tiles.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "aethermap"
SOURCE = CACHE / "sam_road"
CITYSCALE = ROOT / "cityscale_vitb_512_e10.ckpt"
CITYSCALE_INFERENCE = CACHE / "cityscale_inference_state.pt"
SAM_VIT_B = CACHE / "sam_vit_b_01ec64.pth"
CITYSCALE_SHA256 = "f1369a067d3cf3126324457fb80e499a74ea126dc23df94eef502e7c1fe51445"
CITYSCALE_INFERENCE_SHA256 = "4a5df0ce08333e91b35bc7d022deb5117d26d29f7a607b85e4201f3cf11337f3"
SAM_VIT_B_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
_net = None
_config = None
_device = None


def availability() -> tuple[bool, str]:
    if not (SOURCE / "model.py").is_file() or not (SOURCE / "sam" / "segment_anything" / "modeling" / "image_encoder.py").is_file():
        return False, "Original SAM-Road code and submodules are missing"
    if not (SOURCE / "config" / "toponet_vitb_512_cityscale.yaml").is_file():
        return False, "CityScale configuration is missing"
    slim_ready = CITYSCALE_INFERENCE.is_file() and CITYSCALE_INFERENCE.stat().st_size == 351_352_571
    original_ready = CITYSCALE.is_file() and CITYSCALE.stat().st_size == 1_054_078_423
    if not slim_ready and not original_ready:
        return False, "CityScale trained checkpoint is missing"
    if not SAM_VIT_B.is_file() or SAM_VIT_B.stat().st_size != 375_042_383:
        return False, "SAM ViT-B base checkpoint is missing"
    return True, "ready"


def _check_sha(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint failed SHA-256 verification: {path.name}")


def load_model():
    global _net, _config, _device
    if _net is not None:
        return _net, _config, _device
    available, reason = availability()
    if not available:
        raise FileNotFoundError(reason)
    use_slim = CITYSCALE_INFERENCE.is_file() and CITYSCALE_INFERENCE.stat().st_size == 351_352_571
    weights_path = CITYSCALE_INFERENCE if use_slim else CITYSCALE
    _check_sha(weights_path, CITYSCALE_INFERENCE_SHA256 if use_slim else CITYSCALE_SHA256)
    _check_sha(SAM_VIT_B, SAM_VIT_B_SHA256)

    import torch
    import yaml

    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from model import SAMRoad

    settings = yaml.safe_load((SOURCE / "config" / "toponet_vitb_512_cityscale.yaml").read_text())
    settings["SAM_CKPT_PATH"] = str(SAM_VIT_B)
    config = SimpleNamespace(**settings)
    with contextlib.redirect_stdout(io.StringIO()):
        net = SAMRoad(config)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint if use_slim else checkpoint["state_dict"], strict=True)
    del checkpoint
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    _net, _config, _device = net, config, device
    return _net, _config, _device


def _nms(points, scores, radius):
    import numpy as np
    from scipy.spatial import cKDTree

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)
    order = np.argsort(scores)[::-1]
    ordered = points[order]
    tree = cKDTree(ordered)
    retained = np.ones(len(ordered), dtype=bool)
    for i, point in enumerate(ordered):
        if retained[i]:
            retained[tree.query_ball_point(point, radius)] = False
            retained[i] = True
    return ordered[retained]


def _graph_points(keypoint, road, config):
    import numpy as np

    def candidates(mask, threshold, radius):
        ys, xs = np.where(mask > threshold)
        points = np.column_stack((xs, ys)).astype(np.int32)
        scores = mask[ys, xs]
        return _nms(points, scores, radius)

    junctions = candidates(keypoint, config.ITSC_THRESHOLD, config.ITSC_NMS_RADIUS)
    road_points = candidates(road, config.ROAD_THRESHOLD, config.ROAD_NMS_RADIUS)
    if len(junctions) == 0:
        return road_points
    if len(road_points) == 0:
        return junctions
    points = np.concatenate((junctions, road_points))
    priority = np.concatenate((np.ones(len(junctions)) * 2, np.zeros(len(road_points))))
    return _nms(points, priority, config.ROAD_NMS_RADIUS)


def _tile_starts(size, patch_size):
    if size == patch_size:
        return [0]
    return list(range(0, size - patch_size, patch_size // 2)) + [size - patch_size]


def predict_road_graph(image):
    """Return mask and image-pixel graph edges for a 512/1024 square PIL image."""
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    net, config, device = load_model()
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    if width != height or width not in (512, 1024):
        raise ValueError("SAM-Road expects a 512 or 1024 pixel square image")
    starts = _tile_starts(width, config.PATCH_SIZE)
    fused = np.zeros((height, width, 2), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    tiles = []

    with torch.inference_mode():
        for y in starts:
            for x in starts:
                patch = torch.from_numpy(pixels[y:y + 512, x:x + 512].copy()).unsqueeze(0).to(device)
                scores, features = net.infer_masks_and_img_features(patch)
                fused[y:y + 512, x:x + 512] += scores[0].cpu().numpy()
                counts[y:y + 512, x:x + 512] += 1
                tiles.append((x, y, features))
        fused /= counts[..., None]
        points = _graph_points(fused[..., 0], fused[..., 1], config)
        if len(points) > 3000:
            points = points[:3000]
        edge_scores = defaultdict(list)
        for x, y, features in tiles:
            eligible = np.where((points[:, 0] >= x) & (points[:, 0] < x + 512) &
                                (points[:, 1] >= y) & (points[:, 1] < y + 512))[0]
            if len(eligible) < 2:
                continue
            local = points[eligible] - np.array([x, y])
            distances, neighbors = cKDTree(local).query(local, k=min(17, len(local)),
                                                          distance_upper_bound=config.NEIGHBOR_RADIUS)
            if neighbors.ndim == 1:
                neighbors = neighbors[:, None]
                distances = distances[:, None]
            pairs = np.zeros((len(local), 16, 2), dtype=np.int64)
            valid = np.zeros((len(local), 16), dtype=bool)
            for i in range(len(local)):
                pairs[i, :, 0] = i
                pairs[i, :, 1] = i
                selected = [int(n) for n, d in zip(neighbors[i], distances[i]) if n < len(local) and n != i and d <= config.NEIGHBOR_RADIUS][:16]
                pairs[i, :len(selected), 1] = selected
                valid[i, :len(selected)] = True
            scores = net.infer_toponet(
                features,
                torch.from_numpy(local).unsqueeze(0).to(device),
                torch.from_numpy(pairs).unsqueeze(0).to(device),
                torch.from_numpy(valid).unsqueeze(0).to(device),
            )[0, :, :, 0].cpu().numpy()
            for i, j in np.argwhere(valid):
                edge = tuple(sorted((int(eligible[i]), int(eligible[pairs[i, j, 1]]))))
                edge_scores[edge].append(float(scores[i, j]))
    del tiles
    edges = [(points[i].tolist(), points[j].tolist(), float(np.mean(values)))
             for (i, j), values in edge_scores.items()
             if np.mean(values) >= config.TOPO_THRESHOLD]
    return fused[..., 1], edges


def graph_paths(edges, min_length=15):
    """Collapse degree-two graph edges into reviewable road polylines."""
    import math
    import networkx as nx

    graph = nx.Graph()
    for start, end, score in edges:
        a, b = tuple(start), tuple(end)
        if a != b:
            graph.add_edge(a, b, score=score)
    visited = set()
    paths = []

    def walk(a, b):
        vertices = [a, b]
        scores = [graph[a][b]["score"]]
        visited.add(frozenset((a, b)))
        previous, current = a, b
        while graph.degree[current] == 2:
            next_node = next(node for node in graph.neighbors(current) if node != previous)
            key = frozenset((current, next_node))
            if key in visited:
                break
            vertices.append(next_node)
            scores.append(graph[current][next_node]["score"])
            visited.add(key)
            previous, current = current, next_node
        length = sum(math.dist(vertices[i - 1], vertices[i]) for i in range(1, len(vertices)))
        if length >= min_length:
            paths.append((vertices, sum(scores) / len(scores)))

    for node in graph.nodes:
        if graph.degree[node] != 2:
            for neighbor in graph.neighbors(node):
                if frozenset((node, neighbor)) not in visited:
                    walk(node, neighbor)
    for a, b in graph.edges:
        if frozenset((a, b)) not in visited:
            walk(a, b)
    return sorted(paths, key=lambda item: -len(item[0]))
