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

    class AnimationControl(object):
        def __init__(self):
            self.paused = True
            self.index = 0

        def onkey(self, event):
            self.paused = not self.paused
            #self.index += 1
            #print(sides[0][self.index])

        def __iter__(self):
            while self.index < len(sides[0]):
                yield self.index
                if not self.paused:
                    print(sides[0][self.index])
                    self.index += 1

    control = AnimationControl()
    figure.canvas.mpl_connect('key_press_event', control.onkey)

    def image(path):
        return imread(path)
        #return preprocess(imread(path))[..., 0]

    canvases = [plotter.imshow(image(paths[0]), cmap=cm.gray) for (plotter, paths) in zip(plotters, sides)]
    def update(i):
        for (data, paths) in zip(canvases, sides):
            data.set_data(image(paths[i]))

        return canvases

    animation = FuncAnimation(figure, update, control, interval=100, repeat=False)

    plt.show()


def main():
    args = arguments()
    show(args)


if __name__ == '__main__':
    main()
