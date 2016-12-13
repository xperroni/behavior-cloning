import csv
from math import ceil
from os.path import isdir

from keras import backend as K
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential

import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import sobel

from inputs import arguments, Balanced, Batches


def save(model, args):
    path = args.path_model
    with open(path + '.json', 'w') as output:
        output.write(model.to_json())

    model.save_weights(path + '.h5')


def Model(input_shape, hidden, breadth, dropout, reach=None):
    nonlinear='relu'

    model = Sequential()
    model.add(Convolution2D(24, 5, 5, name='conv1', subsample=(2, 2), input_shape=input_shape, activation=nonlinear))
    model.add(Convolution2D(36, 5, 5, name='conv2', subsample=(2, 2), activation=nonlinear))
    model.add(Convolution2D(48, 5, 5, name='conv3', subsample=(2, 2), activation=nonlinear))
    model.add(Convolution2D(64, 3, 3, name='conv4', activation=nonlinear))
    model.add(Convolution2D(64, 3, 3, name='conv5', activation=nonlinear))

    model.add(Flatten())
    model.add(Dense(hidden, name='hidden', activation=nonlinear))
    model.add(Dense(breadth, name='output', activation='softmax'))

    half_breadth = breadth // 2
    angle = lambda x: K.cast(K.argmax(x), 'float32') / half_breadth - 1.0
    model.add(Lambda(angle, output_shape=(1,)))

    training = Sequential()
    training.add(model.get_layer('conv1'))
    training.add(model.get_layer('conv2'))
    training.add(model.get_layer('conv3'))
    training.add(model.get_layer('conv4'))
    training.add(model.get_layer('conv5'))
    training.add(Flatten())
    training.add(Dropout(dropout))
    training.add(model.get_layer('hidden'))
    training.add(Dropout(dropout))
    training.add(model.get_layer('output'))

    return (model, training)


def train(args):
    batches = Balanced(args)

    (model, training) = Model(
        (66, 200, 3),
        args.hidden,
        args.breadth,
        args.dropout
    )

    training.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    training.fit_generator(batches,
        samples_per_epoch=len(batches),
        nb_epoch=args.epochs
    )

    return model


def main():
    args = arguments()
    model = train(args)
    save(model, args)


if __name__ == '__main__':
    main()
