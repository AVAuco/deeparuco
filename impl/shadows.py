import sys
from math import gcd, lcm
from random import randint, random, uniform

import cv2
import numpy as np
from perlin_numpy import generate_fractal_noise_2d

sys.path.insert(0, "../")

from impl.effects import rotate3d


def gradient(width, height):
    t_size = max(width, height)
    size = t_size * 2

    grad = np.zeros((size, size))

    for i in range(size):
        grad[i] = i / size

    center = grad.shape[0] // 2
    mat = cv2.getRotationMatrix2D((center, center), random() * 360, 1.0)
    pic = cv2.warpAffine(grad, mat, (size, size))

    # Final crop

    center = grad.shape[0] // 2
    pic = pic[
        center - height // 2 : center + height // 2,
        center - width // 2 : center + width // 2,
    ]

    # Re-range

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    return pic


def lines(width, height, num_patterns=3):
    t_size = max(width, height)
    size = t_size * 2

    pic = np.ones((size, size))
    center = pic.shape[0] // 2

    for i in range(num_patterns):
        curr = 0

        while curr < size:
            paint = randint(
                1, max((size - curr) // 2, 1)
            )  # min(randint(0, 16), size - curr)
            skip = randint(
                1, max((size - curr - paint) // 2, 1)
            )  # min(randint(0, 16), size - curr - paint)
            pic[curr : curr + paint] *= uniform(0.0, 2.0)  # random()
            curr = curr + paint + skip

        # Rotate

        mat = cv2.getRotationMatrix2D((center, center), random() * 360, 1.0)
        pic = cv2.warpAffine(pic, mat, (pic.shape[0], pic.shape[1]))

    # Re-range

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    # Perspective

    pic = cv2.merge((pic, pic, pic, np.ones(pic.shape))) * 255.0
    pic, _ = rotate3d(pic, randint(-30, 30), randint(-30, 30), 0)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) / 255.0

    # Final crop

    center = pic.shape[0] // 2
    pic = pic[
        center - height // 2 : center + height // 2,
        center - width // 2 : center + width // 2,
    ]

    return pic


def circular(width, height):
    pic = np.zeros((height, width))
    center = (randint(0, height), randint(0, width))

    diag = int((width**2 + height**2) ** (1 / 2))

    radius = randint(diag // 4, diag)

    for i in range(height):
        for j in range(width):
            pic[i, j] = max(
                1 - (((i - center[0]) ** 2 + (j - center[1]) ** 2) ** (1 / 2) / radius),
                0,
            )

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    return pic


def perlin(width, height, bins=0, octaves=4):
    t_width = lcm(width, 2 ** (octaves - 1))
    t_height = lcm(height, 2 ** (octaves - 1))

    res_x = t_width // gcd(t_width, t_height)
    res_y = t_height // gcd(t_width, t_height)

    # Fractal noise

    pic = generate_fractal_noise_2d((t_height, t_width), (res_y, res_x), octaves)

    # Re-range

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    # Threshold

    if bins > 1:
        pic = np.digitize(pic, [(i + 1) / bins for i in range(bins - 1)]) / (bins - 1)
    return pic