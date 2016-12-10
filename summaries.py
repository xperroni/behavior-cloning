from model import arguments, load_dataset


def summaries(args):
    dataset = load_dataset(args.path_datasets, args.breadth, culled=False)
    breadth = dataset.y.breadth
    spread = dataset.y.spread
    classes = dataset.y.classes
    print(spread)
    for i in range(breadth):
        t = 2 * (i / (breadth - 1) - 0.5) * spread
        print('Class %d (%.3f rad): %d items' % (i, t, len(classes[i])))


def main():
    args = arguments()
    summaries(args)


if __name__ == '__main__':
    main()
