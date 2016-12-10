import csv
from math import sin, cos

import matplotlib.pyplot as plt

import numpy as np


ANGLE = 3
SPEED = 6

def load(path):
    X = []
    Y = []
    s = np.array([[0.0, 0.0]]).T
    t = 0
    with open(path) as stream:
        data = csv.reader(stream)
        for row in data:
            l = np.array([[float(row[SPEED]), 0]]).T
            t += float(row[ANGLE]) * 0.245
            R = np.array([
                [cos(t), sin(t)],
                [-sin(t), cos(t)]
            ])

            s += R.dot(l)

            (x, y) = s.flat
            X.append(x)
            Y.append(y)

    return (X, Y)


def plot(path):
    (X, Y) = load(path)
    plt.plot(X, Y, 'b-')
    plt.show()


def main():
    from sys import argv
    plot(argv[1])


if __name__ == '__main__':
    main()
