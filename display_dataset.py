from glob import glob
from os.path import join as joinpath

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from scipy.misc import imread

from inputs import arguments, preprocess


def list_paths(folder, prefix):
    paths = glob(joinpath(folder, 'IMG', prefix + '_*.jpg'))
    paths.sort()
    return paths


def show(args):
    folder = args.path_datasets[0]
    sides = [
        list_paths(folder, 'left'),
        list_paths(folder, 'center'),
        list_paths(folder, 'right')
    ]

    (figure, plotters) = plt.subplots(1, 3)

    def image(path):
        return preprocess(imread(path))[..., 0]

    #data = plotter.imshow(splice(imread(paths[0])), animated=True)
    canvases = [plotter.matshow(image(paths[0]), cmap=cm.gray) for (plotter, paths) in zip(plotters, sides)]
    def update(i):
        print(i)

        for (data, paths) in zip(canvases, sides):
            data.set_data(image(paths[i]))
        return canvases

    animation = FuncAnimation(figure, update, len(sides[0]), interval=100, repeat=False)

    plt.show()


def main():
    args = arguments()
    show(args)


if __name__ == '__main__':
    main()
