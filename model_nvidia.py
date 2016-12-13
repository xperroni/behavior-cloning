import csv
from math import ceil
from os.path import isdir

from keras import backend as K
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential

import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import sobel

from inputs import arguments, preprocess
from datasets import resample, Dataset, Images, Data


CENTER_IMAGE = 0
LEFT_IMAGE = 1
RIGHT_IMAGE = 2
STEERING_ANGLE = 3
THROTLE = 4

def load_dataset(args):
    paths = args.path_datasets
    breadth = args.breadth
    padded = args.padded

    if len(paths) == 1 and isdir(paths[0]):
        return Dataset(paths[0])

    def image(path):
        return preprocess(imread(path.strip()))

    half_breadth = breadth // 2
    angle_step = 0.01

    def truncate(value, a, b):
        return min(max(value, a), b)

    X = Images('X.bin')
    y = Data('y.bin')
    for path in paths:
        with open(path) as stream:
            data = csv.reader(stream)
            for row in data:
                if float(row[THROTLE]) < 1:
                    continue

                image_c = image(row[CENTER_IMAGE])
                #image_l = image(row[LEFT_IMAGE])
                image_r = image(row[RIGHT_IMAGE])

                angle = float(row[STEERING_ANGLE])
                label_c = angle
                #label_l = angle + angle_step
                #label_r = angle - angle_step

                #X.extend([image_c, image_l, image_r])
                #y.extend([label_c, label_l, label_r])
                X.append(image_c)
                y.append(label_c)

    return Dataset('./', X, y)


def save(model, args):
    with open(args.path_model, 'w') as output:
        output.write(model.to_json())

    model.save_weights(args.path_weights)


def Model(input_shape):
    nonlinear='tanh'

    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=input_shape, activation=nonlinear))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation=nonlinear))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation=nonlinear))
    model.add(Convolution2D(64, 3, 3, activation=nonlinear))
    model.add(Convolution2D(64, 3, 3, activation=nonlinear))

    model.add(Flatten())

    model.add(Dense(1164, name='hidden1', activation=nonlinear))
    model.add(Dense(100, name='hidden2', activation=nonlinear))
    model.add(Dense(50, name='hidden3', activation=nonlinear))
    model.add(Dense(10, name='hidden4', activation=nonlinear))
    model.add(Dense(1, name='output'))

    return model


def train(args):
    D_train = load_dataset(args)

    model = Model(D_train.X.shape[1:])

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(D_train.X.data, D_train.y.data,
        batch_size=args.batch,
        nb_epoch=args.epochs,
        verbose=1
    )

    return model


def main():
    args = arguments()
    model = train(args)
    save(model, args)


if __name__ == '__main__':
    main()
