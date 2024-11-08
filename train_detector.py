from argparse import ArgumentParser
from os import remove

from ultralytics import YOLO

if __name__ == '__main__':

    parser = ArgumentParser(description = 'DeepArUco++ detector trainer.')
    parser.add_argument('source_dir', help = 'where to find source images')
    parser.add_argument('run_name', help = 'directory of the resulting model')
    parser.add_argument('--model', '-m', help = 'base model to train', default='yolov8m')
    args = parser.parse_args()

    with open(f'{args.run_name}.yaml', 'w') as f:
        f.write(f'path: \'{args.source_dir}\'\n')
        f.write('train: \'train/images\'\n')
        f.write('val: \'valid/images\'\n')
        f.write('names:\n  0: \'marker\'')

    model = YOLO(f'models/{args.model}.pt')
    model.train(data = f'{args.run_name}.yaml',
                rect = True, iou = 0.5, 
                batch = -1, 
                epochs = 1000, patience = 10, 
                cache = True,
                name = args.run_name)

    remove(f'{args.run_name}.yaml')