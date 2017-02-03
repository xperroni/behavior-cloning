# Behavioral Cloning - End-to-end Deep Learning Autonomous Steering

This project demonstrates the use of convolutional networks and deep learning to perform *behavioral cloning*, that is, the learning and reproduction of a (presumably human) driver's responses to different road conditions. Two different neural network architectures &ndash; one that performs [logistic classification](https://en.wikipedia.org/wiki/Multinomial_logistic_regression), the other [nonlinear regression](https://en.wikipedia.org/wiki/Nonlinear_regression) &ndash; were implemented and trained on recordings of a car being manually driven over a simulated environment, then connected to the simulator so they would themselves steer the virtual car. The next sections give more details on the implemented architectures, training methodology and results.

## Implementation & Usage

The project is implemented in Python, on top of the [Keras](https://keras.io/) framework with [TensorFlow](https://www.tensorflow.org/) backend. Code is organized in two modules, the model trainer ([`model.py`](model.py)) and the simulator client ([`drive.py`](drive.py)).

To download training and validation datasets, open a terminal window and type:

    (cd datasets ; ./download.sh)

To run the model trainer, type:

    python model.py <list of train recordings> <validation recording> --architecture <architecture>

For example:

    python model.py middle.csv left.csv right.csv validation.csv --architecture regression

To run the model trainer on training records `middle.csv`, `left.csv` and `right.csv`, with validation record `validation.csv` and using the nonlinear regression architecture.

Once the model is trained, start the simulator and run the client:

    python drive.py model_regression.json

If you don't already have the simulator, follow the steps below to download and start it:

    cd simulator/
    ./download.sh
    ./simulator

To see extra configuration options, type:

    python model.py --help
    python drive.py --help

Alternatively to running the Python modules directly, you can use the following shell scripts to run the model trainer:

    ./train_classification.sh
    ./train_regression.sh

And likewise, for the simulator client:

    ./test_classification.sh
    ./test_regression.sh

## Development

At first the same approach of starting with a minimum working system then iteratively improving performance, same architectural motif of a shallow convolutional network followed by a multilayer perceptron classifier (which here associated camera inputs to discrete angle "classes"), and substantial amounts of code (especially for data loading and management) from the [previous project](https://github.com/xperroni/traffic-signs/) were tried on the steering problem. Soon, however, two problems became apparent:

1. The larger inputs of the steering problem quickly overwhelmed the development machine's memory &ndash; attempts to solve this by using [filesystem-backed arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html) incurred significant processing delays;
2. Whereas in the previous project a very simple architecture trained on non-curated data already proved sufficient to deliver moderately good recognition results, in this case no amount of fiddling with hyperparameters, addition or partitioning of data succeeded in produce behavior that wasn't static or random; steering commands would either send the car straight out of the road or leave it drifting away slowly but surely.

The first problem was the easiest to solve, by dropping the borrowed data management code and starting over with a generator implementation fit for use with Keras. The generator keeps a list of filesystem paths to recorded images, loading (and unloading) them one batch at a time. Despite the overhead of reloading images multiple times, should they be selected for more than one batch, the improvements in memory usage more than make up for this relative to the previous solution.

The steeper minimal requirements for successful steering were achieved in two steps. First, the deeper convolutional network proposed in the so-called [NVIDIA paper](doc/nvidia.pdf) was incorporated in the system. Second, training data was curated and partitioned so that different modes of driving (through a straight road, turning left, right and returning to the middle lane from the borders) would be exposed to the training network in equal proportion. Once the system achieved satisfactory performance levels, the full architecture described in the paper was implemented for comparison.

## Architectures

Two architectures were implemented in this project. The first uses a convolutional network to extract features from input images, then feeds them into a logistic classification network that maps inputs to a set of discrete steering angles (represented as cells of one-hot encoded vectors). Rectified Linear Units (ReLU's) are used to introduce nonlinearity between layers. Once the network is trained, an extra layer converting the vectors back into angle values is plugged at the end. See the figure below for an illustration.

<img src="https://xperroni.github.io/behavior-cloning/images/architecture_classification.svg" width="800">

The second architecture takes the output of the convolutional network and feeds it into a multilayer perceptron network, whose funneling layers converge into a single output value &ndash; the steering angle associated to a given output. In contrast to the previous architecture, the hyperbolic tangent function is used instead of ReLU's for nonlinearity transform; this is due to the fact that the output of this network is in the range `[-1, 1]`, whereas classification network outputs are in the range `[0, 1]`.

<img src="https://xperroni.github.io/behavior-cloning/images/architecture_regression.svg" width="800">

Both networks also perform normalization of visual inputs. This is done by cropping image inputs to (row, column) dimensions `(66, 200)`, then subtracting the mean and dividing by the standard deviation.

## Experiments

Training data was collected in three recording sessions, with distinct driving styles:

1. Trying to keep the car in the middle of the road;
2. Deliberately drifting left, then pulling back into the middle of the road;
3. Deliberately drifting right, then pulling back into the middle of the road.

After recording, data from session 1 (labeled "middle") was split in three sets, corresponding to left turns, right turns and straight sections of the route. These sets would be sampled in equal proportion during training, thus ensuring that inputs from straight driving sessions (which represented the vast majority of records) wouldn't overshadow the others.

Data from sessions 2 ("left recovery")  and 3 ("right recovery") were pruned so that only data relevant for drift recovery was kept. This was done by keeping only those records from session 2 with positive steering angles (i.e. those that showed the car recovering from a left drift), and conversely for session 3, only those with negative angles (that showed a recovery from a right drift). These culled records would also be sampled in the same proportion as those split from session 1.

For each training record, all three visual inputs (from the front, left and right cameras) were used for training. Left camera inputs were associated to the recorded steering angle plus a `0.1` offset, and right camera inputs, the angle minus `0.1`.

A fourth session was recorded in the same driving style as session 1, to be used for validation. No test data was recorded, as testing would be performed through the simulator.

Both architectures displayed successful steering across the simulated track 1, however the classification architecture drove much more smoothly, especially over straight sections, where the regression architecture displayed a tendency to wobble left and right.
