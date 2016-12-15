import csv
from argparse import ArgumentParser
from math import ceil
from random import sample

from keras import backend as K
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential

import numpy as np
from scipy.misc import imread
from skimage.color import rgb2hsv


def arguments():
    r'''Parse command-line arguments.
    '''
    parser = ArgumentParser(description='Model generation and training')

    parser.add_argument('path_datasets', nargs='+', type=str, help='List of paths to training datasets.')
    parser.add_argument('--architecture', type=str, default='regression', help='Architecture to use, one of ("classification", "regression").')
    parser.add_argument('--reach', type=float, default=0.5, help='Maximum absolute steering angle possible.')
    parser.add_argument('--breadth', type=int, default=21, help='Encoding resolution of the steering angle vector.')
    parser.add_argument('--hidden', type=int, default=1164, help='Number of hidden elements in the fully-connected module.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Fraction of randomly selected layer inputs to drop during training.')
    parser.add_argument('--batch', type=int, default=18, help='Minimum size of training batches, increased if not a multiple of `sum(len(datasets)) * 3`.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs in the training step.')
    parser.add_argument('--path_model', type=str, default='model', help='Path to model architecture.')

    return parser.parse_args()


def truncate(value, a, b):
    r'''Return `value` if within range `(a, b)` (inclusive),
        otherwise return the closest range limit value.
    '''
    return min(max(value, a), b)


def continuous_y(angle, breadth, offset=0):
    r'''Return the angle as a continuous value, possibly modified by an offset.
    '''
    step = 1.0 / (breadth // 2)
    return angle + step * offset


def onehot_y(angle, breadth, offset=0):
    r'''Return the angle as a one-hot vector of length `breadth`.

        The position of the non-zero value may be displaced by `offset` positions.
        Negative values cause it to shift left, and positive ones, right.
    '''
    encoded = np.zeros(breadth)
    half_breadth = breadth // 2
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


CENTER_IMAGE = 0
LEFT_IMAGE = 1
RIGHT_IMAGE = 2
STEERING_ANGLE = 3
THROTLE = 4
BREAK = 5
SPEED = 6

class Batches(object):
    r'''
    '''
    def __init__(self, encoder_y, **kwargs):
        self.breadth = kwargs['breadth']
        self.encoder_y = encoder_y
        self.datasets = []

        for path in kwargs['path_datasets']:
            with open(path) as stream:
                dataset = []
                for row in csv.reader(stream):
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

                    dataset.append(row)

                self.datasets.append(dataset)

        # Each dataset sample is composed of three entries, corresponding to center,
        # left and right images, hence the multiplication factor.
        k = len(self.datasets) * 3
        batch_size = kwargs['batch']
        self.samples_per_dataset = int(ceil(batch_size / k))

        n = sum(len(dataset) for dataset in self.datasets)
        d = self.samples_per_dataset * k
        self.__len = n - (n % d)


    def __next__(self):
        breadth = self.breadth
        encoder_y = self.encoder_y

        def encoder_x(path):
            return imread(path.strip())

        X = []
        y = []
        for row in self.sample():
            image_c = encoder_x(row[CENTER_IMAGE])
            image_l = encoder_x(row[LEFT_IMAGE])
            image_r = encoder_x(row[RIGHT_IMAGE])

            angle = float(row[STEERING_ANGLE])
            label_c = encoder_y(angle, breadth)
            label_l = encoder_y(angle, breadth, 1)
            label_r = encoder_y(angle, breadth, -1)

            X.extend([image_c, image_l, image_r])
            y.extend([label_c, label_l, label_r])

        return (np.array(X), np.array(y))

    def __len__(self):
        return self.__len

    def sample(self):
        r'''Randomly selects an equal number of samples from each dataset, so that
            total entry number sums up to the batch size.
        '''
        return sum((sample(dataset, self.samples_per_dataset) for dataset in self.datasets), [])


class Model(object):
    r'''Basic network architecture class. Implements input normalization and convolution
        layers, leaving the definition of connected layers to subclasses.
    '''
    def __init__(self, name, input_shape, nonlinear, encoder_y, **kwargs):
        self.name = name
        self.path = kwargs['path_model'] + '_' + name
        self.data = Batches(encoder_y, **kwargs)
        self.epochs = kwargs['epochs']

        # It's often needed to use a different layer set for training than for testing
        # (e.g. when using dropout layers). Therefore Model encapsulates two Keras models,
        # adding layers to either or both as needed.
        self.graph = Sequential()
        self.training = Sequential()

        def normalize(x):
            r'''Normalize input batches to mean 0 and standard deviation 1, then crops
                height and width to dimensions (66, 200).
            '''
            (m, n) = K.int_shape(x)[1:3]

            a = 60
            b = a + 66
            c = (n - 200) // 2
            d = 200 + c

            x = x[:, a:b, c:d, :]
            x -= K.mean(x, keepdims=True)
            x /= K.std(x, keepdims=True)

            return x

        # Convolution layers and parameters were taken from the "nvidia paper" on end-to-end autonomous steering.
        # See docs/nvidia.pdf
        self.add(Lambda(normalize, input_shape=input_shape, output_shape=(66, 200, 3)))
        self.add(Convolution2D(24, 5, 5, name='conv1', subsample=(2, 2), activation=nonlinear))
        self.add(Convolution2D(36, 5, 5, name='conv2', subsample=(2, 2), activation=nonlinear))
        self.add(Convolution2D(48, 5, 5, name='conv3', subsample=(2, 2), activation=nonlinear))
        self.add(Convolution2D(64, 3, 3, name='conv4', activation=nonlinear))
        self.add(Convolution2D(64, 3, 3, name='conv5', activation=nonlinear))

    def add(self, layer):
        self.graph.add(layer)
        self.training.add(layer)

    def train(self):
        self.training.fit_generator(
            self.data,
            samples_per_epoch=len(self.data),
            nb_epoch=self.epochs
        )

    def save(self):
        with open(self.path + '.json', 'w') as output:
            output.write(self.graph.to_json())

        self.graph.save_weights(self.path + '.h5')


class Classification(Model):
    def __init__(self, input_shape, **kwargs):
        nonlinear = 'relu'
        Model.__init__(self, 'classification', input_shape, nonlinear, onehot_y, **kwargs)

        hidden = kwargs['hidden']
        breadth = kwargs['breadth']
        dropout = kwargs['dropout']

        self.add(Flatten())
        self.training.add(Dropout(dropout))
        self.add(Dense(hidden, name='hidden', activation=nonlinear))
        self.training.add(Dropout(dropout))
        self.add(Dense(breadth, name='outputs', activation='softmax'))

        reach = kwargs['reach']
        half_breadth = kwargs['breadth'] // 2
        angle = lambda x: K.clip(K.cast(K.argmax(x), 'float32') / half_breadth - 1.0, -reach, reach)
        self.graph.add(Lambda(angle, output_shape=(1,)))

        self.training.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


class Regression(Model):
    def __init__(self, input_shape, **kwargs):
        nonlinear = 'tanh'
        Model.__init__(self, 'regression', input_shape, nonlinear, continuous_y, **kwargs)

        dropout = kwargs['dropout']

        self.add(Flatten())
        self.training.add(Dropout(dropout))
        self.add(Dense(1164, name='hidden1', activation=nonlinear))
        self.training.add(Dropout(dropout))
        self.add(Dense(100, name='hidden2', activation=nonlinear))
        self.training.add(Dropout(dropout))
        self.add(Dense(50, name='hidden3', activation=nonlinear))
        self.training.add(Dropout(dropout))
        self.add(Dense(10, name='hidden4', activation=nonlinear))
        self.training.add(Dropout(dropout))
        self.add(Dense(1, name='output', activation=nonlinear))

        self.training.compile(optimizer='adam', loss='mse')


ARCHITECTURES = {
    'classification': Classification,
    'regression': Regression
}


def main():
    args = arguments()
    Architecture = ARCHITECTURES[args.architecture]
    model = Architecture((160, 320, 3), **vars(args))
    model.train()
    model.save()


if __name__ == '__main__':
    main()
