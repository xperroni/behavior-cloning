from model import arguments, load_dataset


def summaries(args):
    dataset = load_dataset(args)
    breadth = dataset.y.breadth
    classes = dataset.y.classes
    for i in range(breadth):
        n = len(classes[i])
        t = 2 * (i / (breadth - 1) - 0.5)
        print('Class %d (%.3f rad): %d items' % (i, t, n))


def main():
    args = arguments()
    summaries(args)


if __name__ == '__main__':
    main()
