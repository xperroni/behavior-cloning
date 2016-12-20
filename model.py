import csv
from argparse import ArgumentParser
from math import ceil
from random import sample

from keras import backend as K
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np
from scipy.misc import imread
from skimage.color import rgb2hsv


def arguments():
    r'''Parse command-line arguments.
    '''
    parser = ArgumentParser(description='Network generation and training')

    parser.add_argument('path_datasets', nargs='+', type=str, help='List of paths to training datasets. The last entry is withheld for validation.')
    parser.add_argument('--architecture', type=str, default='regression', help='Architecture to use, one of ("classification", "regression").')
    parser.add_argument('--reach', type=float, default=0.5, help='Maximum absolute steering angle possible.')
    parser.add_argument('--breadth', type=int, default=21, help='Encoding resolution of the steering angle vector.')
    parser.add_argument('--hidden', type=int, default=1164, help='Number of hidden elements in the fully-connected module.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Fraction of randomly selected layer inputs to drop during training.')
    parser.add_argument('--batch', type=int, default=18, help='Minimum size of training batches, increased if not a multiple of `total_len(datasets) * 3`.')
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
    r'''Dataset manager for training and validation.
    '''
    def __init__(self, encoder_y, **kwargs):
        r''' Create a new dataset manager.

            **Arguments:**

            `encoder_y`
                Function that transforms raw angles into the representation used by the network.
                Must be of form `encoder_y(angle, breadth, offset=0)`, where `angle` is an input
                steering angle, `breadth` is the number of distinct angle values recognized,
                and `offset` is an optional offset from the recognized value.

            **Named parameters:**

            `breadth`
                Number of distinct angle values recognized by the network.

            `epochs`
                Number of epochs run in training.

            `path_datasets`
                Sequence of paths to dataset CSV files. The last entry is withheld for validation.

            `batch`
                Minimum number of entries per batch. This may be increased if it's not a multiple of
                the number of samples per dataset; see the documentation of `generator()` for details.
        '''
        self.encoder_y = encoder_y
        self.breadth = kwargs['breadth']
        self.epochs = kwargs['epochs']

        def load(path):
            dataset = []
            with open(path) as file:
                for row in csv.reader(file):
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

            return dataset

        # Load training and validation datasets.
        paths = kwargs['path_datasets']
        paths_training = paths[:-1]
        path_validation = paths[-1]
        self.training_datasets = [load(path) for path in paths_training]
        self.validation_dataset = load(path_validation)

        # Compute a number of rows to be sampled from each training dataset, such that the total
        # sums up to (at least) the given batch size. Each dataset row is composed of three samples
        # (corresponding to center, left and right images), hence the multiplication factor in k.
        batch_size = kwargs['batch']
        k = len(self.training_datasets) * 3
        self.rows_per_dataset = int(ceil(batch_size / k))

        # Recalculate the batch size to be a multiple of the number of samples per dataset
        self.batch_size = self.rows_per_dataset * k

        # Compute the number of training samples per epoch as a multiple of the batch size.
        n = sum(len(dataset) for dataset in self.training_datasets)
        self.samples_per_epoch = n - (n % self.batch_size)

        # Compute the number of validation samples per epoch as a multiple of the batch size.
        n = len(self.validation_dataset)
        self.nb_val_samples = n - (n % self.batch_size)

    def generator(self, datasets, rows_per_dataset, samples_per_row=3):
        r'''Create a new generator to iteratively extract samples from the given datasets.

            For each iteration the generator will return `len(datasets) * rows_per_dataset * samples_per_row`
            samples, randomly selected from each dataset in equal proportion. this makes possible to balance
            data sampling against dataset deficiencies, i.e. if a given data class is underrepresented in the
            training dataset, moving its samples to their own separate dataset will cause them to be sampled
            in the same proportion as the rest of the data.

            One side-effect of this arrangement is that batch size must be redefined in terms of the parameters
            above; this is why the batch size may need to be increased from the instantiation parameter.

            **Arguments:**

            `datasets`
                List of datasets to be sampled.

            `rows_per_dataset`
                Number of rows to be sampled from each dataset.

            `samples_per_row`
                Number of samples to be computed from each row. Currently it only makes sense
                to set this argument to `3` (which selects the center, left and right images)
                or `1` (which selects just the center image).
        '''
        breadth = self.breadth
        encoder_y = self.encoder_y

        def encoder_x(path):
            return imread(path.strip())

        # Randomly select an equal number of samples from each dataset, so that
        # total number sums up to the batch size.
        def samples():
            return sum((sample(dataset, rows_per_dataset) for dataset in datasets), [])

        while True:
            X = []
            y = []
            for row in samples():
                image_c = encoder_x(row[CENTER_IMAGE])
                image_l = encoder_x(row[LEFT_IMAGE])
                image_r = encoder_x(row[RIGHT_IMAGE])
                images = [image_c, image_l, image_r]

                angle = float(row[STEERING_ANGLE])
                label_c = encoder_y(angle, breadth)
                label_l = encoder_y(angle, breadth, 1)
                label_r = encoder_y(angle, breadth, -1)
                labels = [label_c, label_l, label_r]

                X.extend(images[0:samples_per_row])
                y.extend(labels[0:samples_per_row])

            yield (np.array(X), np.array(y))

    def training(self):
        r'''Return a generator to iterate over training data.
        '''
        return self.generator(self.training_datasets, self.rows_per_dataset, 3)

    def validation(self):
        r'''Return a generator to iterate over validation data.
        '''
        return self.generator([self.validation_dataset], self.batch_size, 1)


class Network(object):
    r'''Basic network architecture class. Implements input normalization and convolution
        layers, leaving the definition of connected layers to subclasses.
    '''
    def __init__(self, name, input_shape, nonlinear, encoder_y, **kwargs):
        r'''Create a new basic network model.

            Also creates a `Batches` dataset manager to load and deliver training / validation data.

            **Arguments:**

            `name`
                Network name.

            `input_shape
                Tuple describing the dimensions of input images. For 3-channel images this
                should be of the form `(rows, columns, channels)`.

            `nonlinear`
                Nonlinear activation function placed between network layers.

            `encoder_y`
                Function to encode raw angles, see the documentation of `Batches` for details.

            **Named parameters:**

            `path_model`
                Path to output model files up to a name suffix. Final file names will be
                constructed by appending the network's name and appropriate file extensions
                (`.json`, `.h5`).

            Also see the documentation for the `Batches` class, as any named parameters given
            here are also passed to the data manager instance created by this class.
        '''
        self.name = name
        self.path = kwargs['path_model'] + '_' + name
        self.data = Batches(encoder_y, **kwargs)

        # It's often needed to use a different layer set for training than for testing
        # (e.g. when using dropout layers). Therefore the network encapsulates two Keras
        # models, adding layers to either or both as needed.
        self.model = Sequential()
        self.training = Sequential()

        def normalize(x):
            r'''Crops height and width to dimensions (66, 200), then normalize values to
                mean 0 and standard deviation 1.
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
        r'''Add he given layer to both the canonical and training versions of the model.
        '''
        self.model.add(layer)
        self.training.add(layer)

    def train(self):
        r'''Train the network on the loaded datasets.
        '''
        self.training.fit_generator(
            self.data.training(),
            samples_per_epoch=self.data.samples_per_epoch,
            validation_data=self.data.validation(),
            nb_val_samples=self.data.nb_val_samples,
            nb_epoch=self.data.epochs
        )

    def save(self):
        r'''Save the network structure and weights to the path given at instantiation.
        '''
        with open(self.path + '.json', 'w') as output:
            output.write(self.model.to_json())

        self.model.save_weights(self.path + '.h5')


class Classification(Network):
    r'''Steering network based on classification.
    '''
    def __init__(self, input_shape, **kwargs):
        r'''Create a new classification network.

            **Arguments:**

            `input_shape
                Tuple describing the dimensions of input images. For 3-channel images this
                should be of the form `(rows, columns, channels)`.

            **Named parameters:**

            `hidden`
                Number of cells in the hidden layer after the convolution stack.

            `breadth`
                Number of cells in the logit layer.

            `reach`
                Maximum absolute steering angle returned by the network.

            `dropout`
                Dropout rate.

            `path_model`
                Path to output model files up to a name suffix. Final file names will be
                constructed by appending the network's name and appropriate file extensions
                (`.json`, `.h5`).

            Also see the documentation for the `Batches` class, as any named parameters given
            here are also passed to the data manager instance created by this class.
        '''
        nonlinear = 'relu'
        Network.__init__(self, 'classification', input_shape, nonlinear, onehot_y, **kwargs)

        hidden = kwargs['hidden']
        breadth = kwargs['breadth']
        dropout = kwargs['dropout']

        self.add(Flatten())
        self.training.add(Dropout(dropout))
        self.add(Dense(hidden, name='hidden', activation=nonlinear))
        self.training.add(Dropout(dropout))
        self.add(Dense(breadth, name='outputs', activation='softmax'))

        reach = kwargs['reach']
        half_breadth = breadth // 2
        angle = lambda x: K.clip(K.cast(K.argmax(x), 'float32') / half_breadth - 1.0, -reach, reach)
        self.model.add(Lambda(angle, output_shape=(1,)))

        self.training.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


class Regression(Network):
    def __init__(self, input_shape, **kwargs):
        r'''Create a new regression network.

            **Arguments:**

            `input_shape
                Tuple describing the dimensions of input images. For 3-channel images this
                should be of the form `(rows, columns, channels)`.

            **Named parameters:**

            `dropout`
                Dropout rate.

            `path_model`
                Path to output model files up to a name suffix. Final file names will be
                constructed by appending the network's name and appropriate file extensions
                (`.json`, `.h5`).

            Also see the documentation for the `Batches` class, as any named parameters given
            here are also passed to the data manager instance created by this class.
        '''
        nonlinear = 'tanh'
        Network.__init__(self, 'regression', input_shape, nonlinear, continuous_y, **kwargs)

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
