import sys
from argparse import ArgumentParser
from glob import glob
from os.path import basename, dirname
from random import randint, random, seed, uniform
from shutil import copy

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "../")

from impl.shadows import gradient, lines, perlin


def change_lighting(pic, funcs):
    width, height = pic.shape[1], pic.shape[0]
    gmin, gmax = sorted([uniform(0.0, 2.0) for i in range(2)])
    shadows = np.ones((height, width))

    # Apply gradients

    if random() > 0.5:
        shadows *= gradient(width, height)

    # Apply other augmentations

    for f in funcs:
        if random() > 0.5:
            shadows *= f(width, height)

    # Re-range

    shadows = (shadows - np.min(shadows)) / (
        np.max(shadows) - np.min(shadows) + 1e-6
    ) * (gmax - gmin) + gmin

    # Get output

    out = np.clip(np.multiply(pic, np.expand_dims(shadows, -1)), 0, 255)

    return out


def get_augmentations(
    path, funcs, blur=False, noise=False, c_shift=False, num_augmentations=9
):
    pic = cv2.imread(path)
    labels_file = dirname(dirname(path)) + f'/labels/{basename(path).split(".")[0]}.txt'

    pics = [pic.astype(np.float32) for i in range(num_augmentations)]

    if c_shift == True:
        for i in range(len(pics)):
            for c in range(3):
                c_min, c_max = np.min(pics[i][:, :, c]), np.max(pics[i][:, :, c])

                t_min = max(c_min, 8) * uniform(0.0, 2.0)
                t_max = max(t_min + 8, c_max * uniform(0.9, 1.1))

                pics[i][:, :, c] = (
                    np.multiply(
                        np.divide(
                            (pics[i][:, :, c] - c_min),
                            (c_max - c_min),
                            casting="unsafe",
                        ),
                        (t_max - t_min),
                        casting="unsafe",
                    )
                    + t_min
                )

            pics[i] = np.clip(pics[i], 0, 255)

    pics = [change_lighting(pic, funcs) for pic in pics]

    if blur == True:
        for i in range(len(pics)):
            if random() < 0.2:
                k_size = 2 * randint(1, 3) + 1
                pics[i] = cv2.GaussianBlur(pics[i], (k_size, k_size), 0)

    if noise == True:
        for i in range(len(pics)):
            if random() < 0.2:
                sigma = randint(8, 32)
                gaussian = np.random.normal(0, sigma, pics[i].shape)
                pics[i] = np.clip(pics[i] + gaussian, 0, 255)

    for i in range(num_augmentations):
        cv2.imwrite(f"{dirname(path)}/{i + 1}{basename(path)[1:]}", pics[i])
        copy(labels_file, f"{dirname(labels_file)}/{i + 1}{basename(labels_file)[1:]}")


if __name__ == "__main__":
    parser = ArgumentParser(description="DeepArUco v2 data augmentation.")
    parser.add_argument("source_dir", help="where to find source images")
    parser.add_argument(
        "-l",
        "--lines",
        help="use patterns with lines to generate shadows",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--perlin",
        help="use perlin noise to generate shadows",
        action="store_true",
    )
    parser.add_argument(
        "-b", "--blur", help="add full-image gaussian blur", action="store_true"
    )
    parser.add_argument(
        "-n", "--noise", help="add full-image gaussian noise", action="store_true"
    )
    parser.add_argument(
        "-c", "--color_shift", help="add full-image color shift", action="store_true"
    )
    args = parser.parse_args()

    seed(0)
    np.random.seed(0)

    data_dir = args.source_dir

    funcs = []
    if args.lines:
        funcs.append(lines)
    if args.perlin:
        funcs.append(perlin)

    for part in ["train", "valid"]:
        imgs = glob(data_dir + "/" + part + "/images/*.jpg")
        for img in tqdm(imgs):
            get_augmentations(img, funcs, args.blur, args.noise, args.color_shift)
