#!/usr/bin/env python

# Show image

import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt


def show_default(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (640, 480))
    print('Showing.\nPress ESC to exit...')
    cv2.imshow('Image', image)
    cv2.waitKey()


def read_calibration_parameters(path):
    converters = {
        'baseline': lambda line: float(line.split('=')[1]),
        'cam': lambda line: float(line.split('=')[1].strip('[').strip(']').strip(';').split(' ')[0]),
        'doffs': lambda line: float(line.split('=')[1]),
    }
    params = {
        'baseline': None,  # camera baseline
        'cam': None,  # focal length
        'doffs': None  # x-difference of principal points
    }
    with open(path, 'r') as calib_file:
        lines = calib_file.readlines()
        for line in lines:
            line = line.strip()
            for key in params:
                if line.startswith(key):
                    params[key] = converters[key](line)
    params['f'] = params['cam']
    del params['cam']
    return params


def show_depth(path):
    # folder = os.path.dirname(os.path.abspath(path))
    # calib_file = os.path.join(folder, 'calib.txt')
    # calib = read_calibration_parameters(calib_file)
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    # def d2z(d): return calib['baseline'] * calib['f'] / (d + calib['doffs'])
    # image = np.array(map(d2z, image))
    # print('Using the following calibration parameters: ' + str(calib))
    # image = cv2.resize(image, (640, 480))
    print(image.shape)
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: ' + sys.argv[0] + ' IMAGE')
        sys.exit(1)

    print('Reading ' + sys.argv[1] + '...')

    if sys.argv[1].endswith('.pfm'):
        show_depth(sys.argv[1])
    else:
        show_default(sys.argv[1])
