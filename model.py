import csv
import pickle

from keras import backend as K
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential

import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import sobel

from inputs import arguments, preprocess
from datasets import split, Tensors, Likelihoods, Culled, Dataset


ANGLE_STEP = 0.2

CENTER_IMAGE = 0
LEFT_IMAGE = 1
RIGHT_IMAGE = 2
STEERING_ANGLE = 3

def load_split(path, breadth):
    dataset = load_dataset(path, breadth)
    spread = dataset.y.spread
    (train, val) = split(dataset)
    train.y.spread = spread

    return (train, val)


def load_dataset(paths, breadth, culled=False):
    if len(paths) == 1 and paths[0].endswith('.p'):
        return load_pickled(paths[0])

    X = []
    y = []
    for path in paths:
        with open(path) as stream:
            data = csv.reader(stream)
            for row in data:
                X.append(preprocess(imread(row[CENTER_IMAGE].strip())))
                X.append(preprocess(imread(row[LEFT_IMAGE].strip())))
                X.append(preprocess(imread(row[RIGHT_IMAGE].strip())))

                angle = float(row[STEERING_ANGLE])
                y.append(angle)
                y.append(angle + ANGLE_STEP)
                y.append(angle - ANGLE_STEP)

    X = Tensors(np.array(X))

    y = np.array(y)
    y_max = np.abs(y).max()

    half_breadth = breadth // 2
    step = y_max / half_breadth

    classes = [None] * breadth
    classes[half_breadth] = (y == 0)
    for i in range(half_breadth):
        y_minus = ((i * -step) > y) & (y > ((i + 1) * -step))
        y_plus  = ((i *  step) < y) & (y < ((i + 1) *  step))

        classes[half_breadth - 1 - i] = y_minus
        classes[half_breadth + 1 + i] = y_plus

    for (i, k) in enumerate(classes):
        y[k] = i

    y = Likelihoods(y.astype(np.int), breadth)
    y.spread = y_max

    dataset = Dataset('Train', X, y)
    if culled:
        dataset = Culled(dataset)

    save_pickled(dataset, 'dataset.p')

    return dataset


def load_pickled(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_pickled(dataset, path):
    with open(path, 'wb') as file:
        pickle.dump(dataset, file)


def save(trained, test, args):
    with open(args.path_model, 'w') as output:
        output.write(test.to_json())

    trained.save_weights(args.path_weights)


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
    (D_train, D_val) = load_split(args.path_datasets, args.breadth)

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
        reach=D_train.y.spread
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
