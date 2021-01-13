import argparse
import json
import logging
import os
import random
import shutil
from timeit import default_timer as timer

import cv2
import numpy as np

from single_shot.processing import SingleShotProcessing


def process_initial_dirs(video_name, save_dir, experiment_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exp_dir = os.path.join(save_dir, video_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_dir = os.path.join(exp_dir, experiment_name)
    os.environ['EXP_DIR'] = exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    os.environ['RES_DIR'] = os.path.join(exp_dir, experiment_name)
    if not os.path.exists(os.environ['RES_DIR']):
        os.makedirs(os.environ['RES_DIR'])


def main(arguments):
    capture = cv2.VideoCapture(arguments.video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.environ['img_w'] = str(width)
    os.environ['img_h'] = str(height)
    os.environ['VIDEO_NAME'] = arguments.video_name
    os.environ['exp_name'] = arguments.exp_name
    if arguments.save_dir is not None:
        os.environ['save_dir'] = arguments.save_dir
    else:
        os.environ['save_dir'] = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(os.environ['save_dir']):
        os.makedirs(os.environ['save_dir'])

    timer0 = timer()
    with open(arguments.detections, 'r') as clean__:
        detections = json.load(clean__)
        detections = {int(k): v for k, v in detections.items()}
    process_initial_dirs(arguments.video_name, os.environ['save_dir'], arguments.exp_name)

    logging.basicConfig(filename=os.path.join(os.environ['RES_DIR'], 'info.log'), level=logging.DEBUG)
    proc = SingleShotProcessing(detections, arguments.start_frame, arguments.path_to_homography_dict)
    new_meta = proc.process()
    with open(os.path.join(os.environ['RES_DIR'], 'single_shot_try.json'), 'w') as json__:
        json.dump(new_meta, json__)
    logging.info('all time {} '.format(timer() - timer0))


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    py_cache_path = os.path.join(dir_path, '__pycache__')
    try:
        shutil.rmtree(py_cache_path)
    except BaseException:
        print('Error while deleting directory')

    parser = argparse.ArgumentParser(
        description='custom arguments without using Sacred library ')
    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    parser.add_argument('--detections')
    parser.add_argument('--video_path')
    parser.add_argument('--video_name', type=str)
    parser.add_argument('--exp_name')
    parser.add_argument('--path_to_homography_dict')
    parser.add_argument('--start_frame', default=None, type=int)
    parser.add_argument('--save_dir', type=int)


    args = parser.parse_args()
    main(args)
