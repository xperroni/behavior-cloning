import csv


STEERING_ANGLE = 3

def select(direction, path_in, path_out):
    u = 1.0 if direction == 'right' else -1.0

    with open(path_in, newline='') as inputs, open(path_out, 'w', newline='') as outputs:
        data = csv.reader(inputs)
        out = csv.writer(outputs)
        for row in data:
            if float(row[STEERING_ANGLE]) * u > 0:
                out.writerow(row)


def main():
    from sys import argv
    select(*argv[1:])


if __name__ == '__main__':
    main()
