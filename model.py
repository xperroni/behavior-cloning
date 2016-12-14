import csv
from math import ceil
from os.path import isdir

from keras import backend as K
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential

import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import sobel

from inputs import arguments, continuous_y, onehot_y, Balanced


#def save(model, args):
    #path = args.path_model
    #with open(path + '.json', 'w') as output:
        #output.write(model.to_json())

    #model.save_weights(path + '.h5')


#def Model(input_shape, dropout):
    #nonlinear='tanh'

    #model = Sequential()
    #model.add(Convolution2D(24, 5, 5, name='conv1', subsample=(2, 2), input_shape=input_shape, activation=nonlinear))
    #model.add(Convolution2D(36, 5, 5, name='conv2', subsample=(2, 2), activation=nonlinear))
    #model.add(Convolution2D(48, 5, 5, name='conv3', subsample=(2, 2), activation=nonlinear))
    #model.add(Convolution2D(64, 3, 3, name='conv4', activation=nonlinear))
    #model.add(Convolution2D(64, 3, 3, name='conv5', activation=nonlinear))

    #model.add(Flatten())
    #model.add(Dense(1164, name='hidden1', activation=nonlinear))
    #model.add(Dense(100, name='hidden2', activation=nonlinear))
    #model.add(Dense(50, name='hidden3', activation=nonlinear))
    #model.add(Dense(10, name='hidden4', activation=nonlinear))
    #model.add(Dense(1, name='output', activation=nonlinear))

    #training = Sequential()
    #training.add(model.get_layer('conv1'))
    #training.add(model.get_layer('conv2'))
    #training.add(model.get_layer('conv3'))
    #training.add(model.get_layer('conv4'))
    #training.add(model.get_layer('conv5'))
    #training.add(Flatten())
    #training.add(Dropout(dropout))
    #training.add(model.get_layer('hidden1'))
    #training.add(Dropout(dropout))
    #training.add(model.get_layer('hidden2'))
    #training.add(Dropout(dropout))
    #training.add(model.get_layer('hidden3'))
    #training.add(Dropout(dropout))
    #training.add(model.get_layer('hidden4'))
    #training.add(Dropout(dropout))
    #training.add(model.get_layer('output'))

    #return (model, training)


#def train(args):
    #batches = Balanced(args, continuous_y)

    #(model, training) = Model((66, 200, 3), args.dropout)

    #training.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
    #training.fit_generator(batches,
        #samples_per_epoch=len(batches),
        #nb_epoch=args.epochs
    #)

    #return model


class Model(object):
    def __init__(self, name, encoder_y, **kwargs):
        self.name = name
        self.model = Sequential()
        self.training = Sequential()
        self.path = kwargs['path_model'] + '_' + name
        self.data = Balanced(encoder_y, **kwargs)
        self.epochs = kwargs['epochs']

    def add(self, layer):
        self.model.add(layer)
        self.training.add(layer)

    def init_convolutions(self, input_shape, nonlinear):
        self.add(Convolution2D(24, 5, 5, name='conv1', subsample=(2, 2), input_shape=input_shape, activation=nonlinear))
        self.add(Convolution2D(36, 5, 5, name='conv2', subsample=(2, 2), activation=nonlinear))
        self.add(Convolution2D(48, 5, 5, name='conv3', subsample=(2, 2), activation=nonlinear))
        self.add(Convolution2D(64, 3, 3, name='conv4', activation=nonlinear))
        self.add(Convolution2D(64, 3, 3, name='conv5', activation=nonlinear))

    def train(self):
        self.training.fit_generator(
            self.data,
            samples_per_epoch=len(self.data),
            nb_epoch=self.epochs
        )

    def save(self):
        with open(self.path + '.json', 'w') as output:
            output.write(self.model.to_json())

        self.model.save_weights(self.path + '.h5')


class Classification(Model):
    def __init__(self, input_shape, **kwargs):
        Model.__init__(self, 'classification', onehot_y, **kwargs)

        nonlinear = 'relu'
        hidden = kwargs['hidden']
        breadth = kwargs['breadth']
        dropout = kwargs['dropout']

        self.init_convolutions(input_shape, nonlinear)

        self.add(Flatten())
        self.training.add(Dropout(dropout))
        self.add(Dense(hidden, name='hidden', activation=nonlinear))
        self.training.add(Dropout(dropout))
        self.add(Dense(breadth, name='outputs', activation='softmax'))

        self.training.compile(optimizer='adam', loss='categorical_cross_entropy', metrics=['accuracy'])


class Regression(Model):
    def __init__(self, input_shape, **kwargs):
        Model.__init__(self, 'regression', continuous_y, **kwargs)

        nonlinear = 'tanh'
        dropout = kwargs['dropout']

        self.init_convolutions(input_shape, nonlinear)

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

        self.training.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])


ARCHITECTURES = {
    'classification': Classification,
    'regression': Regression
}


def main():
    args = arguments()
    Architecture = ARCHITECTURES[args.architecture]
    model = Architecture((66, 200, 3), **vars(args))
    model.train()
    model.save()


if __name__ == '__main__':
    main()
