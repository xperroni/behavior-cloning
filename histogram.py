import csv

from model import arguments


STEERING_ANGLE = 3

def histogram(path):
    y = {}
    with open(path) as stream:
        data = csv.reader(stream)
        for row in data:
            s = float('%.1f' % float(row[STEERING_ANGLE]))
            y[s] = y.get(s, 0) + 1

    angles = list(y.keys())
    angles.sort()

    for a in angles:
        print('%f: %d' % (a, y[a]))


def main():
    args = arguments()
    histogram(args.path_datasets[0])


if __name__ == '__main__':
    main()
