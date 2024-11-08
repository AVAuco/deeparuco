import json
from argparse import ArgumentParser
from glob import glob
from os import mkdir
from os.path import basename, exists
from shutil import rmtree

import cv2
from tqdm import tqdm

import sys

sys.path.insert(0, '../')

from impl.utils import ordered_corners

if __name__ == '__main__':

    parser = ArgumentParser(description = 'FlyingArUco v2 regression dataset builder.')
    parser.add_argument('source_dir', help = 'where to find source images')
    parser.add_argument('annotations_dir', help = 'where to find annotations for source images')
    parser.add_argument('output_dir', help = 'where to store regression dataset')
    parser.add_argument('-a', '--add', help = 'whether to add files to existing dataset', action = 'store_true')
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    if args.add == False:
        if exists(output_dir):
            rmtree(output_dir)
        mkdir(output_dir)

    crop_size = 64
    r_int = lambda x: int(round(x))

    for part in ["train","valid"]:
        with open(f'{output_dir}/{part}.csv', 'w') as f:
            f.write('pic,c1_x,c1_y,c2_x,c2_y,c3_x,c3_y,c4_x,c4_y,rot,id\n')

        pics = glob(f'{args.source_dir}/{part}/images/*.jpg')

        for path in tqdm(pics):
            img = cv2.imread(path)

            with open(f'{args.annotations_dir}/0{basename(path).split(".")[0][1:]}.json', 'r') as f:
                markers = json.load(f)['markers']
            
            for i in range(len(markers)):

                marker = markers[i]

                x = [c[0] for c in marker['corners']]
                y = [c[1] for c in marker['corners']]

                minx, maxx = min(x), max(x)
                miny, maxy = min(y), max(y)

                # Expand bbox

                minx, miny, maxx, maxy = [max(minx - (0.2 * (maxx - minx) + 0.5), 0),
                                        max(miny - (0.2 * (maxy - miny) + 0.5), 0),
                                        min(maxx + (0.2 * (maxx - minx) + 0.5), img.shape[1] - 1),
                                        min(maxy + (0.2 * (maxy - miny) + 0.5), img.shape[0] - 1)]
            
                # Crop marker

                crop = cv2.resize(img[r_int(miny):r_int(maxy),
                                      r_int(minx):r_int(maxx)], (64, 64))

                # Save files

                crop_name = f'{basename(path).split(".")[0]}_{i:02d}.jpg'
                cv2.imwrite(f'{args.output_dir}/{crop_name}', crop)

                x = [(val - minx) / (maxx - minx) for val in x]
                y = [(val - miny) / (maxy - miny) for val in y]
                c = ordered_corners(x, y)

                rot = marker['rot']
                id = marker['id']

                with open(f'{output_dir}/{part}.csv', 'a') as f:
                    f.write(f'{crop_name},{c[0]},{c[1]},{c[2]},{c[3]},{c[4]},{c[5]},{c[6]},{c[7]},{rot},{id}\n')