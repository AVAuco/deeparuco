from math import cos, radians, sin

import cv2
import numpy as np


def rotate3d(pic, rot_x, rot_y, rot_z, f_mult=1.0, fill_color=(0, 0, 0)):
    height, width = [(2 * i) for i in pic.shape[0:2]]

    pic_exp = np.zeros((height, width, 4), dtype=np.uint8)
    pic_exp[:, :, :3] = fill_color
    pic_exp[
        pic.shape[0] // 2 : (height + pic.shape[0]) // 2,
        pic.shape[1] // 2 : (width + pic.shape[1]) // 2,
        :,
    ] = pic

    alpha = radians(rot_x)
    beta = radians(rot_y)
    gamma = radians(rot_z)

    f = (width / 2) * f_mult

    # 2d -> 3d
    proj2d3d = np.asarray(
        [[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 0], [0, 0, 1]]
    )

    # Rotation matrices
    rx = np.asarray(
        [
            [1, 0, 0, 0],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha), cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )

    ry = np.asarray(
        [
            [cos(beta), 0, sin(beta), 0],
            [0, 1, 0, 0],
            [-sin(beta), 0, cos(beta), 0],
            [0, 0, 0, 1],
        ]
    )

    rz = np.asarray(
        [
            [cos(gamma), -sin(gamma), 0, 0],
            [sin(gamma), cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Translation
    T = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, f], [0, 0, 0, 1]])

    # 3d -> 2d
    proj3d2d = np.asarray([[f, 0, width / 2, 0], [0, f, height / 2, 0], [0, 0, 1, 0]])

    # Combine all
    transform = proj3d2d @ (T @ ((rx @ ry @ rz) @ proj2d3d))
    pic_exp = cv2.warpPerspective(
        pic_exp,
        transform,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_color,
    )

    return pic_exp, transform
