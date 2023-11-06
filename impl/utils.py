import warnings

import cv2
import numpy as np
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


def match_rois(gt_markers, pred_markers):
    pred_unmatched = pred_markers.copy()
    pred_matched = [None] * len(gt_markers)

    max_iou = [0.0] * len(gt_markers)

    while pred_unmatched:
        pred = pred_unmatched.pop()
        ious = [IoU(pred, gt) for gt in gt_markers]

        for match in np.argsort(ious)[::-1]:
            if ious[match] > 0.5 and ious[match] > max_iou[match]:
                max_iou[match] = ious[match]

                if pred_matched[match] != None:
                    pred_unmatched.append(pred_matched[match])

                pred_matched[match] = pred
                break

    assert len(pred_markers) >= len([m for m in pred_matched if m != None])

    return pred_matched


dist = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** (1 / 2)


def match_corners(gt_corners, pred_corners):
    pred_unmatched = pred_corners.copy()
    pred_matched = [None] * 4

    distance = [float("inf")] * 4

    while pred_unmatched:
        pred = pred_unmatched.pop()
        dists = [dist((gt["x"], gt["y"]), (pred["x"], pred["y"])) for gt in gt_corners]

        for match in np.argsort(dists):
            if dists[match] < distance[match]:
                distance[match] = dists[match]

                if pred_matched[match] != None:
                    pred_unmatched.append(pred_matched[match])

                pred_matched[match] = pred
                break

    assert len(pred_corners) == len([c for c in pred_matched if c != None])

    return pred_matched
