from argparse import ArgumentParser
from glob import glob
from os import mkdir
from os.path import basename, exists
from shutil import copy, rmtree

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    parser = ArgumentParser(description = 'Background image filtering.')
    parser.add_argument('source_dir', help = 'where to find source images')
    parser.add_argument('output_dir', help = 'where to store generated images + labels')
    args = parser.parse_args()

    if exists(args.output_dir):
        rmtree(args.output_dir)
    mkdir(args.output_dir)

    t_width = 640
    t_height = 360
    num_bins = 10
    target_size = 2500

    paths = glob(f'{args.source_dir}/*.jpg')

    f_paths = []
    bright = []

    for img in tqdm(paths):
        pic = cv2.imread(img)

        if pic.shape[0] > pic.shape[1]:
            pic = cv2.rotate(pic, cv2.ROTATE_90_CLOCKWISE)

        if pic.shape[1] >= t_width and pic.shape[0] >= t_height:
            centerw, centerh = pic.shape[1]//2, pic.shape[0]//2
            pic = pic[centerh - t_height//2:centerh + t_height//2, centerw - t_width//2:centerw + t_width//2]

            f_paths.append(img)
            bright.append(np.median(cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)) / 255.0)

    d = pd.DataFrame(data = {'path': f_paths, 'brightness': bright})
    d['brightness'] = pd.qcut(x = d['brightness'], q = num_bins, labels = [str(i + 1) for i in range(num_bins)])
    d = d.groupby('brightness', group_keys = False).apply(lambda x: x.sample(int(target_size / num_bins), random_state=0))

    for path in d['path']:
        copy(path, f'{args.output_dir}/{basename(path)}')

    d.to_csv(f'{args.output_dir}/brightness.csv', index = False)