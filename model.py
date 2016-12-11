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
from datasets import resample, Dataset, Images, Labels


ANGLE_STEP = 0.1

CENTER_IMAGE = 0
LEFT_IMAGE = 1
RIGHT_IMAGE = 2
STEERING_ANGLE = 3
THROTLE = 4

def load_split(path, breadth):
    dataset = load_dataset(path, breadth)
    spread = dataset.y.spread
    (train, val) = split(dataset)
    train.y.spread = spread

    return (train, val)


def load_dataset(args):
    paths = args.path_datasets
    breadth = args.breadth
    padded = args.padded

    if len(paths) == 1 and isdir(paths[0]):
        return Dataset(paths[0])

    def image(path):
        return preprocess(imread(path.strip()))

    half_breadth = breadth // 2
    def onehot(angle):
        encoded = np.zeros(breadth)
        if angle < -1.0:
            encoded[0] = 1
        elif angle == 0:
            encoded[half_breadth] = 1
        elif angle > 1.0:
            encoded[-1] = 1
        elif angle < 0:
            i = int(ceil((angle + 1.0) * (half_breadth - 1)))
            encoded[i] = 1
        else: # angle > 0
            i = half_breadth + int(ceil(angle * half_breadth))
            encoded[i] = 1

        return encoded

    X = Images('X.bin')
    y = Labels('y.bin', breadth)
    for path in paths:
        with open(path) as stream:
            data = csv.reader(stream)
            for row in data:
                if float(row[THROTLE]) < 1:
                    continue

                angle = float(row[STEERING_ANGLE])

                image_c = image(row[CENTER_IMAGE])
                label_c = onehot(angle)

                X.append(image_c)
                y.append(label_c)

                #image_l = image(row[LEFT_IMAGE])
                #label_l = onehot(angle + ANGLE_STEP)

                #image_r = image(row[RIGHT_IMAGE])
                #label_r = onehot(angle - ANGLE_STEP)

                #X.extend([image_c, image_l, image_r])
                #y.extend([label_c, label_l, label_r])

    dataset = Dataset('./', X, y)
    if padded:
        dataset = resample(dataset)

    return dataset


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
    #(D_train, D_val) = load_split(args.path_datasets, args.breadth)
    D_train = load_dataset(args)

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
        reach=1.0
    )

    trained.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    trained.fit(D_train.X.data, D_train.y.data,
        batch_size=args.batch,
        nb_epoch=args.epochs,
        verbose=1 #,
        #validation_data=(D_val.X.data, D_val.y.data)
    )

    return (trained, test)


def main():
    args = arguments()
    #show(args)
    (trained, test) = train(args)
    save(trained, test, args)


if __name__ == '__main__':
    main()
