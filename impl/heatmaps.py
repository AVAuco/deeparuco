from os.path import dirname
from sys import argv

import cv2
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


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


if __name__ == "__main__":
    data = pd.read_csv(argv[1])
    row = data.sample(ignore_index=True)
    crop = cv2.imread(f'{dirname(argv[1])}/{row["pic"].values[0]}')

    size = 64

    x_vals = [
        row["c1_x"].values[0] * size,
        row["c2_x"].values[0] * size,
        row["c3_x"].values[0] * size,
        row["c4_x"].values[0] * size,
    ]

    y_vals = [
        row["c1_y"].values[0] * size,
        row["c2_y"].values[0] * size,
        row["c3_y"].values[0] * size,
        row["c4_y"].values[0] * size,
    ]

    hmaps = pos_to_heatmap(x_vals, y_vals, size)
    x_map, y_map = pos_from_heatmap(hmaps)

    print(x_map, x_vals)
    print(y_map, y_vals)

    for x, y in zip(x_map, y_map):
        crop[int(y), int(x), :] = (0, 255, 0)

    show_map = np.zeros((64, 64, 3))
    show_map[:, :, 0] = hmaps[:, :, 0] * 255
    show_map[:, :, 1] = hmaps[:, :, 1] * 255 + hmaps[:, :, 3] * 255
    show_map[:, :, 2] = hmaps[:, :, 2] * 255 + hmaps[:, :, 3] * 255
    cv2.imwrite("map.png", show_map)

    cv2.imwrite("test.png", crop)
    cv2.imwrite("map.png", show_map * 1000.0)
