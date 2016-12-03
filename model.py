import csv
from argparse import ArgumentParser
from glob import glob
from os.path import join as joinpath
from time import sleep

from keras import backend as K
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

import numpy as np
from scipy.misc import imread

from datasets import split, Tensors, Likelihoods, Dataset


def grayscale(image):
    grays = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return grays[..., None]


def splice(image):
    (m, n) = image.shape[:2]
    a = m // 2
    b = 3 * m // 4

    lc = 0
    ld = n // 4
    rc = 3 * ld
    rd = n

    return np.hstack((image[a:b, lc:ld], image[a:b, rc:rd]))


def normalize(image):
    image -= image.mean()
    image /= image.std()
    return image


def preprocess(image):
    return normalize(grayscale(image.astype(np.float32)))


CENTER_IMAGE = 0
STEERING_ANGLE = 3

def arguments():
    parser = ArgumentParser(description='Model generation and training')

    parser.add_argument('path_dataset', type=str, help='Path to training data base folder.')
    parser.add_argument('--breadth', type=int, default=7, help='Encoding resolution of the steering angle vector.')
    parser.add_argument('--side', type=int, default=5, help='Length of the side of convolution layers.')
    parser.add_argument('--depth', type=int, default=32, help='Number of output channels for convolution layers.')
    parser.add_argument('--pool', type=int, default=2, help='Length of the side of max-pooling layers.')
    parser.add_argument('--hidden', type=int, default=49, help='Length of the side of max-pooling layers.')
    parser.add_argument('--batch', type=int, default=16, help='Size of training batches.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs in the training step.')
    parser.add_argument('--path_model', type=str, default='model.json', help='Path to model architecture.')
    parser.add_argument('--path_weights', type=str, default='model.h5', help='Path to model weights.')

    return parser.parse_args()


def load(path, breadth):
    X = []
    y = []
    with open(path) as stream:
        data = csv.reader(stream)
        for row in data:
            X.append(preprocess(imread(row[CENTER_IMAGE])))
            y.append(float(row[STEERING_ANGLE]))

    X = Tensors(np.array(X))

    y = np.array(y)
    y_max = np.abs(y).max()
    y += y_max
    y *= (breadth - 1) / (2 * y_max)
    y = np.round(y).astype(np.int)
    y = Likelihoods(y, breadth)

    dataset = Dataset('Train', X, y)
    (train, val) = split(dataset)
    return (train, val, y_max)


def save(trained, test, args):
    with open(args.path_model, 'w') as output:
        output.write(test.to_json())

    trained.save_weights(args.path_weights)


def show(args):
    paths = glob(joinpath(args.path_dataset, 'IMG', 'center_*.jpg'))
    paths.sort()

    (figure, plotter) = plt.subplots(1, 1)

    #data = plotter.imshow(splice(imread(paths[0])), animated=True)
    data = plotter.matshow(splice(imread(paths[0])), cmap=cm.gray)
    def update(path):
        print(path)
        image = splice(imread(path))
        data.set_data(image)
        return data

    animation = FuncAnimation(figure, update, paths, interval=100, repeat=False)

    plt.show()


def Model(inputs, side, depth, pool, hidden, breadth, reach=None):
    model = Sequential()
    model.add(MaxPooling2D(input_shape=inputs, pool_size=(pool, pool)))
    model.add(Convolution2D(depth, side, side, input_shape=inputs, activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    model.add(Convolution2D(depth, side, side, activation='relu'))

    if reach == None:
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(hidden, name='hidden1', activation='relu'))

    if reach == None:
        model.add(Dropout(0.5))

    model.add(Dense(breadth, name='output', activation='softmax'))

    if reach != None:
        angle = lambda x: 2.0 * (K.cast(K.argmax(x), 'float32') / (breadth - 1.0) - 0.5) * reach
        model.add(Lambda(angle, output_shape=(1,)))

    return model


def train(args):
    (D_train, D_val, y_max) = load(args.path_dataset, args.breadth)

    trained = Model(
        D_train.X.shape[1:],
        args.side,
        args.depth,
        args.pool,
        args.hidden,
        args.breadth
    )

    test = Model(
        D_train.X.shape[1:],
        args.side,
        args.depth,
        args.pool,
        args.hidden,
        args.breadth,
        reach=y_max
    )

    trained.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    trained.fit(D_train.X.data, D_train.y.data,
        batch_size=args.batch,
        nb_epoch=args.epochs,
        verbose=1,
        validation_data=(D_val.X.data, D_val.y.data)
    )

    return (trained, test)


def main():
    args = arguments()
    #show(args)
    (trained, test) = train(args)
    save(trained, test, args)


if __name__ == '__main__':
    main()
