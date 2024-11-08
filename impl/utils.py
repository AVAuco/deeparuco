import numpy as np
import cv2
import warnings

from shapely.geometry import Polygon


def ordered_corners(x_vals, y_vals):

    cx, cy = np.mean(x_vals), np.mean(y_vals)
    angles = np.arctan2(x_vals - cx, y_vals - cy)
    indices = np.argsort(angles)

    coords = [c for c in zip(x_vals, y_vals)]
    coords = [coords[i] for i in indices]

    return [v for c in coords for v in c]


def marker_from_corners(crop, corners, t_size):
    dst = np.array(
        [[0, 0], [0, t_size - 1], [t_size - 1, t_size - 1], [t_size - 1, 0]]
    ).astype(np.float32)

    c1 = [corners[0] * 63, corners[1] * 63]
    c2 = [corners[2] * 63, corners[3] * 63]
    c3 = [corners[4] * 63, corners[5] * 63]
    c4 = [corners[6] * 63, corners[7] * 63]

    src = np.clip(np.array([c1, c2, c3, c4]), 0, 63).astype(np.float32)
    h = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(crop, h, (t_size, t_size))

    return out


def IoU(a, b):
    warnings.simplefilter(action="ignore", category=RuntimeWarning)

    roi_a = Polygon(
        [
            (a["corners"][0]["x"], a["corners"][0]["y"]),
            (a["corners"][1]["x"], a["corners"][1]["y"]),
            (a["corners"][2]["x"], a["corners"][2]["y"]),
            (a["corners"][3]["x"], a["corners"][3]["y"]),
        ]
    )

    roi_b = Polygon(
        [
            (b["corners"][0]["x"], b["corners"][0]["y"]),
            (b["corners"][1]["x"], b["corners"][1]["y"]),
            (b["corners"][2]["x"], b["corners"][2]["y"]),
            (b["corners"][3]["x"], b["corners"][3]["y"]),
        ]
    )

    try:
        inter = roi_a.intersection(roi_b).area
        union = roi_a.union(roi_b).area

        return inter / union
    except:
        return 0


def match_rois(gt_markers, pred_markers, min_iou=0.5):

    match = [None] * len(gt_markers)
    match_iou = [0.0] * len(gt_markers)

    # Get pref. match for each gt marker.

    for i in range(len(gt_markers)):
        ious = [IoU(pred, gt_markers[i]) for pred in pred_markers]
        idx = np.argmax(ious)
        if ious[idx] > min_iou:
            match[i] = idx
            match_iou[i] = ious[idx]

    # Delete duplicates
    
    for i in range(len(match)):
        if match[i] != None:
            dupes = np.where(match == match[i])[0]
        else:
            continue
        
        if len(dupes) > 1:
            best_dupe = dupes[np.argmax([match_iou[d] for d in dupes])]
            for d in dupes:
                if d != best_dupe:
                    match[d] = None

    # Return non-null matches

    return [pred_markers[idx] if idx != None else None for idx in match]

dist = lambda a, b: ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** (1 / 2)

def match_corners(gt_corners, pred_corners, max_dist=5):

    match = [None] * 4
    match_distance = [float('inf')] * 4

    # Get pref. match for each gt corner.

    for i in range(4):
        dists = [dist(pred, gt_corners[i]) for pred in pred_corners]
        idx = np.argmin(dists)
        if dists[idx] < max_dist:
            match[i] = idx
            match_distance[i] = dists[idx]

    # Delete duplicates
    
    for i in range(len(match)):
        if match[i] != None:
            dupes = np.where(match == match[i])[0]
        else:
            continue
        
        if len(dupes) > 1:
            best_dupe = dupes[np.argmin([match_distance[d] for d in dupes])]
            for d in dupes:
                if d != best_dupe:
                    match[d] = None

    # Return non-null matches

    return [pred_corners[idx] if idx != None else None for idx in match]