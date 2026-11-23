"""Offline single-layer segmentation for registered, stopped RGB-D views.

The maximum-height XY grid follows Song et al., Sensors 26 (2026), 3767,
doi:10.3390/s26123767. The RGB seam test and thresholds are WP3 experiments.
Coordinates in this module are metres unless identified as grid cells.
"""
from __future__ import annotations

import cv2
import numpy as np

CELL = 0.002
OPEN = np.ones((3, 3), np.uint8)


def components(mask, min_area):
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, OPEN)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            yield np.nonzero(labels == idx)


def footprint(xy, dims):
    rect = cv2.minAreaRect(xy.astype(np.float32))
    box = cv2.boxPoints(rect)
    sides = np.sort(rect[1])  # distances between outer sampled centres
    expected = np.sort(dims[:2])
    oversized = bool(np.any(sides > expected * 1.30))
    yaw = None
    if (not oversized and np.all(sides >= 0.68 * expected) and expected[1] > 1.5 * expected[0]
            and sides[1] > 1.5 * sides[0]):
        edges = np.diff(box[:3], axis=0)
        dx, dy = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        yaw = float(np.degrees(np.arctan2(dy, dx)) % 180)
    return {"xy": xy.mean(axis=0).tolist(), "box": box.tolist(),
            "sides_mm": (1000 * sides).round(2).tolist(), "ambiguous": oversized, "yaw_deg_mod180": yaw}


def image_regions(height, inside, xy, dims):
    mask = inside & (height >= 0.68 * dims[2]) & (height <= 1.40 * dims[2])
    return [footprint(xy[rc], dims) for rc in components(mask, 65)]


def grid_regions(height, inside, xy, bounds, dims, pixel_area):
    lo, hi = bounds
    width, rows = np.ceil((hi - lo) / CELL).astype(int)
    grid = np.full((rows, width), -np.inf, np.float32)
    col, row = np.floor((xy[inside] - lo) / CELL).astype(int).T
    np.maximum.at(grid, (row, col), height[inside])
    mask = (grid >= 0.68 * dims[2]) & (grid <= 1.40 * dims[2])
    min_area = int(np.ceil(65 * pixel_area / CELL**2))
    regions = [np.column_stack((c, r)) for r, c in components(mask, min_area)]
    return grid, regions


def cell_xy(cells, lower):
    return lower + (cells + 0.5) * CELL


def top_gray(gray, shape, lower, normal, offset, transform, info, top_height):
    """Sample the RGB image on the fitted top plane, without colouring grid holes black."""
    row, col = np.indices(shape)
    xy = cell_xy(np.column_stack((col.ravel(), row.ravel())), lower)
    rot, trans = transform[:3, :3], transform[:3, 3]
    table_normal = rot @ normal
    z = (top_height - offset + table_normal @ trans - xy @ table_normal[:2]) / table_normal[2]
    optical = (np.column_stack((xy, z)) - trans) @ rot
    uv, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3),
                              np.asarray(info.k).reshape(3, 3), np.asarray(info.d))
    maps = uv.reshape(*shape, 2).astype(np.float32)
    return cv2.remap(gray.astype(np.float32), maps, None, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))


def split_region(cells, gray, lower, dims):
    """Accept a cut only with observed line contrast and plausible child footprints."""
    base = footprint(cell_xy(cells, lower), dims)
    if not base["ambiguous"]:
        return [base]
    box = cv2.boxPoints(cv2.minAreaRect(cells.astype(np.float32)))
    shorter, longer = np.sort(dims[:2]) / CELL
    candidates = []
    for edge in (box[1] - box[0], box[2] - box[1]):
        if np.linalg.norm(edge) < 1:
            continue
        axis = edge / np.linalg.norm(edge)
        across = np.array([-axis[1], axis[0]])
        along = cells @ axis
        for cut in np.arange(along.min() + 0.68 * shorter, along.max() - 0.68 * shorter, 1):
            side = along < cut
            groups = (cells[side], cells[~side])
            # Require one single-block-sized child and enough evidence for the other.
            single = []
            for group in groups:
                sizes = np.sort(cv2.minAreaRect(group.astype(np.float32))[1]) + 1
                if sizes[0] < 0.68 * shorter or sizes[1] < 0.68 * longer:
                    break
                single.append(sizes[0] <= 1.30 * shorter and sizes[1] <= 1.30 * longer)
            if len(single) != 2 or not any(single):
                continue
            # Sample only the shared span with observed surface on both sides.
            nearby = [cells[(along >= cut + a) & (along < cut + b)] @ across
                      for a, b in ((-3, -1), (1, 3))]
            if any(len(q) < 3 for q in nearby):
                continue
            low = max(q.min() for q in nearby) + 1
            high = min(q.max() for q in nearby) - 1
            cross = np.arange(low, high, 1)
            if len(cross) < 0.55 * shorter:
                continue
            # Missing depth along the seam is allowed, but require local support
            # on both flanks at each evaluated position.
            support = np.logical_and.reduce([np.min(abs(cross[:, None] - q), axis=1) < 1.5 for q in nearby])
            if np.mean(support) < 0.75:
                continue
            line = cut * axis + cross[support, None] * across
            coords = (line[:, None, :] + np.array([-2, 0, 2])[None, :, None] * axis).astype(np.float32)
            samples = cv2.remap(gray, coords, None, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
            left, centre, right = samples.T
            contrast = np.maximum(np.minimum(left, right) - centre, abs(right - left))
            supported = float(np.mean(contrast >= 10))
            if supported >= 0.65:
                candidates.append((supported, float(np.nanmedian(contrast)), side))
    if not candidates:
        return [base]
    fraction, median, side = max(candidates, key=lambda v: v[:2])
    children = []
    for group in (cells[side], cells[~side]):
        for result in split_region(group, gray, lower, dims):
            result.setdefault("seams", []).append({"support": round(fraction, 3), "contrast": round(median, 2)})
            children.append(result)
    return children
