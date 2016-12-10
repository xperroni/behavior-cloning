from argparse import ArgumentParser

import numpy as np


def arguments():
    parser = ArgumentParser(description='Model generation and training')

    parser.add_argument('path_datasets', nargs='+', type=str, help='List of paths to training datasets base folders.')
    parser.add_argument('--breadth', type=int, default=9, help='Encoding resolution of the steering angle vector.')
    parser.add_argument('--side', type=int, default=5, help='Length of the side of convolution layers.')
    parser.add_argument('--depth', type=int, default=32, help='Number of output channels for convolution layers.')
    parser.add_argument('--pool', type=int, default=2, help='Length of the side of max-pooling layers.')
    parser.add_argument('--hidden', type=int, default=49, help='Length of the side of max-pooling layers.')
    parser.add_argument('--batch', type=int, default=16, help='Size of training batches.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs in the training step.')
    parser.add_argument('--path_model', type=str, default='model.json', help='Path to model architecture.')
    parser.add_argument('--path_weights', type=str, default='model.h5', help='Path to model weights.')

    return parser.parse_args()


def grayscale(image):
    grays = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return grays[..., None]


def clipped(image):
    (m, n) = image.shape[:2]
    a = 60
    b = 130

    return image[a:b]

    #c = 20
    #d = -20

    #return image[a:b, c:d]

    #lc = 0
    #ld = n // 4
    #rc = 3 * ld
    #rd = n

    #return np.hstack((image[a:b, lc:ld], image[a:b, rc:rd]))


def normalize(image):
    image -= image.mean()
    image /= image.std()
    return image


def preprocess(image):
    #return normalize(grayscale(image.astype(np.float)))
    return normalize(grayscale(clipped(image).astype(np.float)))
