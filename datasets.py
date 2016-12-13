from itertools import product
from os.path import join as joinpath
from random import choice, randrange, sample

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def resample(dataset):
    classes = dataset.y.classes.values()
    X = dataset.X
    y = dataset.y

    n_class = np.max([len(cases) for cases in classes])

    folder = dataset.path
    Xp = Images(joinpath(folder, 'Xr.bin'))
    yp = Labels(joinpath(folder, 'yr.bin'), y.breadth)
    for cases in classes:
        n = len(cases)
        if n == 0:
            continue

        k = list(cases)
        k.extend([choice(cases) for i in range(n_class - n)])

        Xp.extend(X[k])
        yp.extend(y[k])

    return Dataset(folder, Xp, yp)


def split(dataset, rate=0.25):
    X = dataset.X
    y = dataset.y

    DataX = X.__class__
    DataY = y.__class__

    (X_train, X_valid, y_train, y_valid) = train_test_split(
        X.data,
        y.data,
        test_size=rate,
        random_state=832289
    )

    breadth = dataset.y.breadth
    return (
        Dataset('Train', DataX(X_train), DataY(y_train, breadth)),
        Dataset('Validate', DataX(X_valid), DataY(y_valid, breadth))
    )


def merge(title, *datasets):
    template = datasets[0]
    X = template.X
    y = template.y

    shape_X = (sum(dataset.X.shape[0] for dataset in datasets),) + X.shape[1:]
    shape_y = (sum(dataset.y.shape[0] for dataset in datasets),) + y.shape[1:]
    breadth = y.breadth

    Merged = template.__class__
    DataX = X.__class__
    DataY = y.__class__

    X = DataX(np.zeros(shape_X))
    y = DataY(np.zeros(shape_y), breadth)

    a = 0
    for dataset in datasets:
        b = a + len(dataset)
        X[a:b] = dataset.X.data
        y[a:b] = dataset.y.data
        a = b

    return Merged(title, X, y)


class Attributes(object):
    r'''Dynamic collection of named attributes.
    '''
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class Data(Attributes):
    r'''Base class for single-type data collection classes.
    '''
    def __init__(self, data, **kwargs):
        self.data = data
        self.dtype = kwargs.get('dtype', np.float32)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __len__(self):
        if isinstance(self.data, str):
            return 0

        return len(self.data)

    def resize(self, reshape):
        if isinstance(self.data, str):
            self.data = np.memmap(self.data, mode='w+', dtype=self.dtype, shape=reshape)

        self.data = np.memmap(self.data.filename, dtype=self.dtype, shape=reshape)

    def append(self, item):
        n = len(self)
        shape = (n + 1,)
        if hasattr(item, 'shape'):
            shape = shape + item.shape

        self.resize(shape)

        k = 0
        if n > 0:
            k = randrange(n)
            self.data[-1] = self.data[k]

        self.data[k] = item

    def extend(self, data):
        for item in data:
            self.append(item)

    def shuffle(self):
        data = self.data
        indexes = range(len(data))
        for k in indexes:
            (i, j) = sample(indexes, 2)
            t = np.copy(data[i])
            data[i] = data[j]
            data[j] = t

    @property
    def shape(self):
        return self.data.shape


class Images(Data):
    r'''Collection of multi-dimensional vectors.
    '''
    def __init__(self, data, **kwargs):
        Data.__init__(self, data, **kwargs)

    def image(self, index):
        x = self[index]
        image = np.zeros(x.shape, dtype=np.uint8)
        for d in range(3):
            channel = np.array(x[:, :, d])
            channel -= channel.min()
            channel *= (255.0 / channel.max())
            image[:, :, d] = channel.astype(np.uint8)

        return image


class Labels(Data):
    r'''Collection of vectors indicating the likelihoods an input belongs to each of a set of classes.
    '''
    def __init__(self, data, breadth, **kwargs):
        Data.__init__(self, data, **kwargs)
        self.breadth = breadth

        if isinstance(self.data, str):
            return

        if len(data.shape) == 1:
            data = data[:, None]

        if data.shape[1] == 1:
            self.data = (np.arange(self.breadth) == data).astype(np.float32)

    def classof(self, index):
        return np.argmax(self[index])

    @property
    def classes(self):
        classes = dict((c, []) for c in range(self.breadth))
        for i in range(len(self)):
            k = self.classof(i)
            classes[k].append(i)

        return classes


class Dataset(Attributes):
    r'''A collection of data cases and associated class identifiers.
    '''
    def __init__(self, *args):
        self.path = args[0]
        if len(args) == 1:
            self.__load(self.path)
        elif len(args) == 3:
            self.X = args[1]
            self.y = args[2]
            self.__save(self.path)
        else:
            raise Exception('Invalid argument list: %s' % str(args))

    def __load(self, folder):
        path = joinpath(folder, 'mapped.txt')
        text = ''.join(line.strip() for line in open(path))
        metadata = eval(text)

        def load(filename, shape_data):
            return np.memmap(filename, dtype=np.float32, mode='r', shape=shape_data)

        (X, y) = [load(*spec) for spec in metadata]
        self.X = Images(X)
        self.y = Data(y)

    def __save(self, folder):
        self.shuffle()
        mapped = [self.X.data, self.y.data]
        with open(joinpath(folder, 'mapped.txt'), 'w') as out:
            metadata = [(data.filename, data.shape) for data in mapped]
            out.write(str(metadata))

        for data in mapped:
            data.flush()

    def __len__(self):
        return len(self.X)

    def __str__(self):
        template = (
            '%s dataset\n'
            'Number of entries: %d\n'
            'Input shape: %s\n'
            'Output shape: %s\n'
            'Number of classes: %d\n'
        )

        return template % (
            self.title,
            len(self),
            str(self.X.shape[1:]),
            str(self.y.shape[1:]),
            self.y.breadth
        )

    def extend(self, dataset):
        self.X.extend(dataset.X.data)
        self.y.extend(dataset.y.data)

    def shuffle(self):
        self.X.shuffle()
        self.y,shuffle()


class Culled(Dataset):
    r'''A dataset where classes are trimmed down to the least common length.
    '''
    def __init__(self, *args, **kwargs):
        Dataset.__init__(self, *args, **kwargs)
        X = self.X
        y = self.y

        classes = y.classes.values()
        n = np.min([len(cases) for cases in classes])

        Xp = []
        yp = []
        for cases in classes:
            k = sample(cases, n)
            Xp.extend(X[k])
            yp.extend(y[k])

        X.data = np.array(Xp)
        y.data = np.array(yp)
