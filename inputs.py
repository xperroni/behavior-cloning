import csv
from argparse import ArgumentParser
from collections import defaultdict
from math import ceil
from random import sample

import numpy as np

from scipy.misc import imread, imresize
from skimage.color import rgb2hsv

def arguments():
    parser = ArgumentParser(description='Model generation and training')

    parser.add_argument('path_datasets', nargs='+', type=str, help='List of paths to training datasets base folders.')
    parser.add_argument('--reach', type=float, default=0.5, help='Maximum absolute steering angle possible.')
    parser.add_argument('--breadth', type=int, default=21, help='Encoding resolution of the steering angle vector.')
    parser.add_argument('--hidden', type=int, default=1164, help='Number of hidden elements in the fully-connected module.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Fraction of randomly selected layer inputs to drop during training.')
    parser.add_argument('--batch', type=int, default=16, help='Size of training batches.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs in the training step.')
    parser.add_argument('--path_model', type=str, default='model', help='Path to model architecture.')

    return parser.parse_args()


def clipped(image):
    (m, n) = image.shape[:2]

    a = 60
    b = a + 66
    c = (n - 200) // 2
    d = -c

    return image[a:b, c:d]


def normalize(image):
    image -= image.mean()
    image /= image.std()
    return image


def preprocess(image):
    return normalize(rgb2hsv(clipped(image)).astype(np.float))


CENTER_IMAGE = 0
LEFT_IMAGE = 1
RIGHT_IMAGE = 2
STEERING_ANGLE = 3
THROTLE = 4
BREAK = 5
SPEED = 6

class Batches(object):
    def __init__(self, args, storage=list):
        self.batch_size = args.batch
        self.breadth = args.breadth
        self.data = storage()
        for (i, path) in enumerate(args.path_datasets):
            with open(path) as stream:
                data = csv.reader(stream)
                for row in data:
                    if float(row[SPEED]) < 1.0:
                        continue

                    row = [
                        row[CENTER_IMAGE],
                        row[LEFT_IMAGE],
                        row[RIGHT_IMAGE],
                        float(row[STEERING_ANGLE]),
                        float(row[THROTLE]),
                        float(row[BREAK]),
                        float(row[SPEED])
                    ]

                    self.append(i, row)

    def __next__(self):
        data = self.sample()

        def image(path):
            return preprocess(imread(path.strip()))

        breadth = self.breadth
        half_breadth = breadth // 2

        def truncate(value, a, b):
            return min(max(value, a), b)

        def onehot(angle, offset=0):
            encoded = np.zeros(breadth)

            i = half_breadth # if angle == 0
            if angle < -1:
                i = 0
            elif angle > 1:
                i = breadth - 1
            elif angle < 0:
                i = int(ceil((half_breadth - 1) * (1.0 + angle)))
            elif angle > 0:
                i = half_breadth + int(ceil(half_breadth * angle))

            i = truncate(i + offset, 0, breadth - 1)

            encoded[i] = 1

            return encoded

        X = []
        y = []
        for row in data:
            angle = float(row[STEERING_ANGLE])

            image_c = image(row[CENTER_IMAGE])
            image_l = image(row[LEFT_IMAGE])
            image_r = image(row[RIGHT_IMAGE])

            label_c = onehot(angle)
            label_l = onehot(angle, 1)
            label_r = onehot(angle, -1)

            X.extend([image_c, image_l, image_r])
            y.extend([label_c, label_l, label_r])

        return (np.array(X), np.array(y))

    def __len__(self):
        n = len(self.data)
        d = self.batch_size * 3  # 3 pictures per input
        return n - (n % d)

    def append(self, i, row):
        self.data.append(row)

    def sample(self):
        return sample(self.data, self.batch_size)


class Balanced(Batches):
    def __init__(self, args):
        Batches.__init__(self, args, lambda: defaultdict(list))

    def __len__(self):
        n = sum(len(data) for data in self.data.values())
        d = self.batch_size * 3 * len(self.data)  # 3 pictures per category
        return n - (n % d)

    def append(self, i, row):
        self.data[i].append(row)

    def sample(self):
        return sum((sample(data, self.batch_size) for data in self.data.values()), [])
