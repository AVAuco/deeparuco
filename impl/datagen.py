# -*- coding: utf-8 -*-

from math import ceil
from random import randint, random, shuffle, uniform

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from .aruco import id_to_bits
from .heatmaps import pos_to_heatmap
from .shadows import circular, gradient, lines, perlin
from .utils import marker_from_corners, ordered_corners

rot_codes = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]


def hflip(pic, corners):
    flipped = corners.copy()
    flipped[[0, 2, 4, 6]] = 1 - flipped[[0, 2, 4, 6]]

    x = flipped[[0, 2, 4, 6]]
    y = flipped[[1, 3, 5, 7]]

    return cv2.flip(pic, 1), np.array(ordered_corners(x, y))


def vflip(pic, corners):
    flipped = corners.copy()
    flipped[[1, 3, 5, 7]] = 1 - flipped[[1, 3, 5, 7]]

    x = flipped[[0, 2, 4, 6]]
    y = flipped[[1, 3, 5, 7]]

    return cv2.flip(pic, 0), np.array(ordered_corners(x, y))


def rotate_corners(pic, corners, n):
    aux = corners.copy()
    rotated = np.zeros(8)

    for i in range(n):
        rotated[[0, 2, 4, 6]] = 1 - aux[[1, 3, 5, 7]]
        rotated[[1, 3, 5, 7]] = aux[[0, 2, 4, 6]]
        aux = rotated.copy()

    x = rotated[[0, 2, 4, 6]]
    y = rotated[[1, 3, 5, 7]]

    return cv2.rotate(pic, rot_codes[n - 1]), np.array(ordered_corners(x, y))


class corner_gen(Sequence):
    def __init__(
        self, img_dataframe, source_dir, batch_size, augment=False, normalize=False
    ):
        self.batch_size = batch_size
        self.augment = augment
        self.normalize = normalize

        self.crops = []
        self.labels = []

        self.lpattern_cache = []

        print("Loading data...")
        for _, row in tqdm(img_dataframe.iterrows(), total=img_dataframe.shape[0]):
            crop = cv2.imread(source_dir + "/" + row["pic"])
            corners = np.array(
                [
                    row["c1_x"],
                    row["c1_y"],
                    row["c2_x"],
                    row["c2_y"],
                    row["c3_x"],
                    row["c3_y"],
                    row["c4_x"],
                    row["c4_y"],
                ]
            )

            if self.normalize == True:
                crop = (crop - np.min(crop)) / (np.max(crop) - np.min(crop) + 1e-9)

            self.crops.append(crop)
            self.labels.append(corners)

        temp = list(zip(self.crops, self.labels))
        shuffle(temp)
        self.crops, self.labels = zip(*temp)

        self.length = ceil(len(self.crops) / self.batch_size)

    def __data_generation(self):
        crops = []
        labels = []

        for i in range(self.iterator, self.iterator + self.batch_size):
            crop = self.crops[i % self.length]
            corners = self.labels[i % self.length]

            if self.augment == True:
                # Rotation, flip

                rotate = randint(0, 3)
                if rotate != 0:
                    crop, corners = rotate_corners(crop, corners, rotate)

                if random() >= 0.5:
                    crop, corners = hflip(crop, corners)

                if random() >= 0.5:
                    crop, corners = vflip(crop, corners)

                if random() >= 0.5:
                    p_num = randint(1, 100000)

                    if p_num > len(self.lpattern_cache):
                        width, height = crop.shape[1], crop.shape[0]
                        gmin, gmax = sorted([uniform(0.0, 2.0) for i in range(2)])

                        shadows = gradient(width, height)

                        for f in [lines, perlin, circular]:
                            if random() > 0.5:
                                shadows *= f(width, height)

                        shadows = (shadows - np.min(shadows)) / (
                            np.max(shadows) - np.min(shadows) + 1e-6
                        ) * (gmax - gmin) + gmin
                        self.lpattern_cache.append(shadows)
                    else:
                        shadows = self.lpattern_cache[p_num - 1]

                    crop = np.clip(np.multiply(crop, np.expand_dims(shadows, -1)), 0, 1)

            crops.append(crop)
            labels.append(corners)

        return np.array(crops), np.array(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.iterator = index * self.batch_size
        crop, corners = self.__data_generation()
        return crop, corners

    def on_epoch_end(self):
        self.iterator = 0


class hmap_gen(Sequence):
    def __init__(
        self, img_dataframe, source_dir, batch_size, normalize=False, augment=False
    ):
        self.batch_size = batch_size
        self.augment = augment
        self.normalize = normalize

        self.crops = []
        self.labels = []

        self.lpattern_cache = []

        print("Loading data...")
        for _, row in tqdm(img_dataframe.iterrows(), total=img_dataframe.shape[0]):
            crop = cv2.imread(source_dir + "/" + row["pic"])

            corners_x = [
                row["c1_x"] * 63,
                row["c2_x"] * 63,
                row["c3_x"] * 63,
                row["c4_x"] * 63,
            ]
            corners_y = [
                row["c1_y"] * 63,
                row["c2_y"] * 63,
                row["c3_y"] * 63,
                row["c4_y"] * 63,
            ]

            hmaps = pos_to_heatmap(corners_x, corners_y, 64)
            hmaps = np.sum(hmaps, -1)

            if self.normalize == True:
                crop = (crop - np.min(crop)) / (np.max(crop) - np.min(crop) + 1e-9)
                hmaps = (hmaps - np.min(hmaps)) / (np.max(hmaps) - np.min(hmaps) + 1e-9)

            self.crops.append(crop)
            self.labels.append(hmaps)

        temp = list(zip(self.crops, self.labels))
        shuffle(temp)
        self.crops, self.labels = zip(*temp)

        self.length = ceil(len(self.crops) / self.batch_size)

    def __data_generation(self):
        crops = []
        labels = []

        for i in range(self.iterator, self.iterator + self.batch_size):
            crop = self.crops[i % self.length].copy()
            hmaps = self.labels[i % self.length]

            if self.augment == True:
                # Rotation, flip

                rotate = randint(0, 3)
                if rotate != 0:
                    crop = np.rot90(crop, rotate)
                    hmaps = np.rot90(hmaps, rotate)

                if random() >= 0.5:
                    crop = np.fliplr(crop)
                    hmaps = np.fliplr(hmaps)

                if random() >= 0.5:
                    crop = np.flipud(crop)
                    hmaps = np.flipud(hmaps)

                # Lighting

                if random() >= 0.5:
                    p_num = randint(1, 100000)

                    if p_num > len(self.lpattern_cache):
                        width, height = crop.shape[1], crop.shape[0]
                        gmin, gmax = sorted([uniform(0.0, 2.0) for i in range(2)])

                        shadows = gradient(width, height)

                        for f in [lines, perlin, circular]:
                            if random() > 0.5:
                                shadows *= f(width, height)

                        shadows = (shadows - np.min(shadows)) / (
                            np.max(shadows) - np.min(shadows) + 1e-6
                        ) * (gmax - gmin) + gmin
                        self.lpattern_cache.append(shadows)
                    else:
                        shadows = self.lpattern_cache[p_num - 1]

                    crop = np.clip(np.multiply(crop, np.expand_dims(shadows, -1)), 0, 1)

            crops.append(crop)
            labels.append(hmaps)

        return np.array(crops), np.array(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.iterator = index * self.batch_size
        crops, hmaps = self.__data_generation()

        return crops, hmaps

    def on_epoch_end(self):
        self.iterator = 0


perturbate = lambda val: min(max(val + random() * 0.1 - 0.05, 0.0), 1.0)


class decoder_gen(Sequence):
    def __init__(
        self, img_dataframe, source_dir, batch_size, augment=True, normalize=False
    ):
        self.batch_size = batch_size
        self.normalize = normalize
        self.augment = augment

        self.crops = []
        self.corners = []
        self.labels = []
        self.orientations = []

        self.lpattern_cache = []

        print("Loading data...")

        for _, row in tqdm(img_dataframe.iterrows(), total=img_dataframe.shape[0]):
            crop = cv2.imread(source_dir + "/" + row["pic"])
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            if self.normalize == True:
                crop = (crop - np.min(crop)) / (np.max(crop) - np.min(crop) + 1e-9)

            corners = np.array(
                [
                    row["c1_x"],
                    row["c1_y"],
                    row["c2_x"],
                    row["c2_y"],
                    row["c3_x"],
                    row["c3_y"],
                    row["c4_x"],
                    row["c4_y"],
                ]
            )
            corners = ordered_corners(corners[[0, 2, 4, 6]], corners[[1, 3, 5, 7]])

            bits = np.array(id_to_bits(row["id"])).reshape((6, 6))

            self.crops.append(crop)
            self.corners.append(corners)
            self.labels.append(bits)
            self.orientations.append(row["rot"])

        temp = list(zip(self.crops, self.corners, self.labels, self.orientations))
        shuffle(temp)
        self.crops, self.corners, self.labels, self.orientations = zip(*temp)

        self.length = ceil(len(self.crops) / self.batch_size)

    def __data_generation(self):
        markers = []
        labels = []

        for i in range(self.iterator, self.iterator + self.batch_size):
            crop = self.crops[i % self.length]
            corners = self.corners[i % self.length]
            bits = self.labels[i % self.length]
            orientation = self.orientations[i % self.length]

            if self.augment == True:
                pert_corners = [perturbate(val) for val in corners]
                marker = marker_from_corners(crop, pert_corners, 32)
            else:
                marker = marker_from_corners(crop, corners, 32)

            if orientation != 0:
                marker = cv2.rotate(marker, rot_codes[orientation - 1])

            if self.augment == True:
                rotate = randint(0, 3)
                if rotate != 0:
                    marker = cv2.rotate(marker, rot_codes[rotate - 1])
                    bits = np.rot90(bits, rotate, (1, 0))

                if random() > 0.5:
                    marker = cv2.flip(marker, 1)
                    bits = np.flip(bits, 1)

                if random() > 0.5:
                    marker = cv2.flip(marker, 0)
                    bits = np.flip(bits, 0)

                # Lighting

                if random() >= 0.5:
                    p_num = randint(1, 100000)

                    if p_num > len(self.lpattern_cache):
                        width, height = marker.shape[1], marker.shape[0]
                        gmin, gmax = sorted([uniform(0.0, 2.0) for i in range(2)])

                        shadows = gradient(width, height)

                        for f in [lines, perlin, circular]:
                            if random() > 0.5:
                                shadows *= f(width, height)

                        shadows = (shadows - np.min(shadows)) / (
                            np.max(shadows) - np.min(shadows) + 1e-6
                        ) * (gmax - gmin) + gmin
                        self.lpattern_cache.append(shadows)
                    else:
                        shadows = self.lpattern_cache[p_num - 1]

                    marker = np.clip(np.multiply(marker, shadows), 0, 1)

            markers.append(marker)
            labels.append(bits)

        return np.array(markers), np.array(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.iterator = index * self.batch_size
        marker, bits = self.__data_generation()

        return marker, bits

    def on_epoch_end(self):
        self.iterator = 0
