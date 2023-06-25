from itertools import repeat
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from sys import argv

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from ultralytics import YOLO

from impl.aruco_codes import id_to_bits

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=256)])

def get_marker(crop, corners, target_size):

    dst = np.array([[0, 0],\
            [target_size - 1, 0],\
            [target_size - 1, target_size - 1],\
            [0, target_size - 1]])

    c1 = [corners[0] * 63, corners[1] * 63]
    c2 = [corners[2] * 63, corners[3] * 63]
    c3 = [corners[4] * 63, corners[5] * 63]
    c4 = [corners[6] * 63, corners[7] * 63]

    src = np.array([c1, c2, c3, c4])
    h, _ = cv2.findHomography(src, dst)
    marker = cv2.warpPerspective(crop, h, (target_size, target_size))
    marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)

    return (marker - np.min(marker)) / (np.max(marker) - np.min(marker) + 1e-9)

ids_as_bits = [id_to_bits(i) for i in range(250)]

def find_id(bits):

    rot0 = bits.flatten()
    rot90 = np.rot90(bits, 1).flatten()
    rot180 = np.rot90(bits, 2).flatten()
    rot270 = np.rot90(bits, 3).flatten()

    distances = [int(np.min([np.sum(np.abs(rot0 - check_bits)),
                np.sum(np.abs(rot90 - check_bits)),
                np.sum(np.abs(rot180 - check_bits)),
                np.sum(np.abs(rot270 - check_bits))])) 
                for check_bits in ids_as_bits]
    
    id = int(np.argmin(distances))

    return (id, distances[id])

pic_path = argv[1]
detector_path = './models/detector.pt'
regressor_path = './models/regressor_mobilenet.h5'
decoder_path = './models/simple_decoder.h5'

detector = YOLO(detector_path)

corner_regressor = load_model(regressor_path)
corner_regressor.trainable = False

marker_decoder = load_model(decoder_path)
marker_decoder.trainable = False

line_width = 2

@tf.function
def refine_corners(corners):
    return corner_regressor(corners)

@tf.function
def decode_markers(markers):
    return marker_decoder(markers)

with Pool(max(1, cpu_count() - 2)) as pool:

    pic = cv2.imread(pic_path)
    detections = detector(pic, conf = 0.03, verbose = False)[0].boxes
    if not detections:
        print('No markers found.')
        quit()

    # Expanded bboxes

    xyxy = [[int(max(det[0] - (0.2 * (det[2] - det[0]) + 0.5), 0)),\
            int(max(det[1] - (0.2 * (det[3] - det[1]) + 0.5), 0)),\
            int(min(det[2] + (0.2 * (det[2] - det[0]) + 0.5), pic.shape[1] - 1)),\
            int(min(det[3] + (0.2 * (det[3] - det[1]) + 0.5), pic.shape[0] - 1))]\
            for det in [[int(val) for val in det.xyxy.cpu().numpy()[0]] for det in detections]]

    crops = [cv2.resize(pic[det[1]:det[3],det[0]:det[2]], (64, 64)) for det in xyxy]
    #for i in range(len(crops)): cv2.imwrite(f'crop_{i}.png', crops[i])

    refined_corners = refine_corners(np.array(crops)).numpy()

    markers = pool.starmap(get_marker, zip(crops, refined_corners, repeat(32)))
    #for i in range(len(markers)): cv2.imwrite(f'marker_{i}.png', markers[i] * 255.0)

    decoded_bits = decode_markers(np.array(markers)).numpy()
    ids = pool.map(find_id, decoded_bits)

    # Bbox only.

    corners = [0,0,1,0,1,1,0,1]
    pic_bbox = pic.copy()

    for det in [[int(val) for val in det.xyxy.cpu().numpy()[0]] for det in detections]:

        width = det[2] - det[0]
        height = det[3] - det[1]

        for i in range(0, 8, 2):
            p1 = (int(det[0] + corners[i] * width), int(det[1] + corners[i + 1] * height))
            i2 = (i + 2) % 8
            p2 = (int(det[0] + corners[i2] * width), int(det[1] + corners[i2 + 1] * height))
            pic_bbox = cv2.line(pic_bbox, p1, p2, (0, 255, 0), line_width, cv2.LINE_AA)

    cv2.imwrite("output_bbox.png", pic_bbox)

    # Refined bbox. only.

    pic_refined = pic.copy()

    for corners, det, id in zip(refined_corners, xyxy, ids):

        color = (0, 255, 0)
            
        width = det[2] - det[0]
        height = det[3] - det[1]

        for i in range(0, 8, 2):
            p1 = (int(det[0] + corners[i] * width), int(det[1] + corners[i + 1] * height))
            i2 = (i + 2) % 8
            p2 = (int(det[0] + corners[i2] * width), int(det[1] + corners[i2 + 1] * height))
            pic_refined = cv2.line(pic_refined, p1, p2, color, line_width, cv2.LINE_AA)

    cv2.imwrite("output_refined.png", pic_refined)

    # Visualize

    thresh = 8
    pic = 2.0 * pic

    for corners, det, id in zip(refined_corners, xyxy, ids):

        color = (0, 255, 0)

        if id[1] >= thresh:
            color = (255, 0, 0)

        width = det[2] - det[0]
        height = det[3] - det[1]

        for i in range(0, 8, 2):
            p1 = (int(det[0] + corners[i] * width), int(det[1] + corners[i + 1] * height))
            i2 = (i + 2) % 8
            p2 = (int(det[0] + corners[i2] * width), int(det[1] + corners[i2 + 1] * height))
            pic = cv2.line(pic, p1, p2, color, line_width, cv2.LINE_AA)

        pic = cv2.putText(pic, str(id[0]), (det[0] + int(width / 2) - 20, det[1] + int(height / 2) + 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line_width, cv2.LINE_AA)

    cv2.imwrite("output.png", pic)
