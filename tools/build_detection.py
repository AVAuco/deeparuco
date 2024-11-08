from argparse import ArgumentParser
from json import load
from os import mkdir
from os.path import basename, dirname, exists
from shutil import copy, rmtree

import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm

if __name__ == '__main__':

    parser = ArgumentParser(description = 'FlyingArUco v2 detection dataset builder.')
    parser.add_argument('source_dir', help = 'where to find source images/labels')
    parser.add_argument('output_dir', help = 'where to store detection dataset')
    parser.add_argument('-a', '--add', help = 'whether to add files to existing dataset', action = 'store_true')
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    train_dir = output_dir + '/train'
    valid_dir = output_dir + '/valid'

    if args.add == False:
        if exists(output_dir):
            rmtree(output_dir)

        mkdir(output_dir)

        mkdir(train_dir)
        mkdir(train_dir + '/images')
        mkdir(train_dir + '/labels')

        mkdir(valid_dir)
        mkdir(valid_dir + '/images')
        mkdir(valid_dir + '/labels')

    width = 640
    height = 360
    validation_size = 0.2

    data = pd.read_csv(source_dir + '/brightness.csv')
    train, valid = train_test_split(data, test_size = 0.2, stratify = data['brightness'], random_state=0)

    train = [f'{source_dir}/{basename(path)}' for path in train['path'].values]
    valid = [f'{source_dir}/{basename(path)}' for path in valid['path'].values]

    for path in tqdm(train):
        copy(path, train_dir + f'/images/{basename(path)}')

        json_file = dirname(path) + f'/{basename(path).split(".")[0]}.json'

        with open(json_file) as file:
            data = load(file)['markers']

        with open(train_dir + f'/labels/{basename(json_file).split(".")[0]}.txt', 'w') as f:
            for marker in data:
                x = [corner[0] for corner in marker['corners']]
                y = [corner[1] for corner in marker['corners']]
                minx, miny, maxx, maxy = [min(x), min(y), max(x), max(y)]
                f.write(f'0 {(minx + maxx) / (2 * width)} {(miny + maxy) / (2 * height)} {(maxx - minx) / width} {(maxy - miny) / height}\n')

    for path in tqdm(valid):
        copy(path, valid_dir + f'/images/{basename(path)}')

        json_file = dirname(path) + f'/{basename(path).split(".")[0]}.json'

        with open(json_file) as file:
            data = load(file)['markers']

        with open(valid_dir + f'/labels/{basename(json_file).split(".")[0]}.txt', 'w') as f:
            for marker in data:
                x = [corner[0] for corner in marker['corners']]
                y = [corner[1] for corner in marker['corners']]
                minx, miny, maxx, maxy = [min(x), min(y), max(x), max(y)]
                f.write(f'0 {(minx + maxx) / (2 * width)} {(miny + maxy) / (2 * height)} {(maxx - minx) / width} {(maxy - miny) / height}\n')
