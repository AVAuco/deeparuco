import cv2
import numpy as np
from scipy.stats import multivariate_normal

# https://stackoverflow.com/a/44947434


def pos_to_heatmap(x_vals, y_vals, size):

    x = np.linspace(0, size - 1, 64)
    y = np.linspace(0, size - 1, 64)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    s = np.eye(2) * 10
    maps = []

    for j, i in zip(x_vals, y_vals):
        m = (i, j)
        k = multivariate_normal(mean=m, cov=s)
        maps.append(k.pdf(xxyy).reshape((size, size)))

    return np.swapaxes(np.array(maps), 0, -1)


def pos_from_heatmap(hmap, detector):

    hmap = (hmap - np.min(hmap)) / (np.max(hmap) - np.min(hmap) + 1e-9)
    keypoints = detector.detect((255 * (1 - hmap)).astype(np.uint8))

    masks = [
        cv2.circle(
            np.zeros(hmap.shape[:2], dtype=np.uint8),
            (int(kp.pt[0]), int(kp.pt[1])),
            int(round(kp.size / 2)),
            255,
            -1,
        )
        for kp in keypoints
    ]
    maps = [cv2.bitwise_and(hmap, hmap, mask=mask) * 255.0 for mask in masks]

    x_vals, y_vals = [], []

    if maps:
        sums = [np.sum(map) for map in maps]
        sorted_idx = np.argsort(sums)[::-1][:4]

        maps, sums = zip(*[(maps[i], sums[i]) for i in sorted_idx])

        r_mat = np.multiply.outer(np.arange(hmap.shape[0]), np.ones(hmap.shape[0]))
        c_mat = r_mat.T

        x_vals, y_vals = zip(
            *[
                (
                    np.sum(np.multiply(map, c_mat)) / (sum * (map.shape[1] - 1)),
                    np.sum(np.multiply(map, r_mat)) / (sum * (map.shape[0] - 1)),
                )
                for map, sum in zip(maps, sums)
            ]
        )

    return x_vals, y_vals


def visualize_hmaps(hmaps, crop):

    crop[:, :, 0] += hmaps[:, :, 0]
    crop[:, :, 1] += hmaps[:, :, 1] + hmaps[:, :, 3]
    crop[:, :, 2] += hmaps[:, :, 2] + hmaps[:, :, 3]

    return crop
