""" test script to verify the CG method works, and time it versus cholesky """

from __future__ import print_function

import argparse
import json
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import scipy.io
import seaborn

from implicit._als import calculate_loss
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

try:
    import implicit.cuda  # noqa
    has_cuda = True
except ImportError:
    has_cuda = False


def benchmark_accuracy(plays):
    output = defaultdict(list)

    def store_loss(model, name):
        def inner(iteration, elapsed):
            loss = calculate_loss(plays, model.item_factors, model.user_factors, 0)
            print("model %s iteration %i loss %.5f" % (name, iteration, loss))
            output[name].append(loss)
        return inner

    for steps in [2, 3, 4]:
        model = AlternatingLeastSquares(factors=100, use_native=True, use_cg=True, regularization=0,
                                        iterations=25)
        model.cg_steps = steps
        model.fit_callback = store_loss(model, 'cg%i' % steps)
        model.fit(plays)

    if has_cuda:
        model = AlternatingLeastSquares(factors=100, use_native=True, use_gpu=True,
                                        regularization=0, iterations=25)
        model.fit_callback = store_loss(model, 'gpu')
        model.use_gpu = True
        model.fit(plays)

    model = AlternatingLeastSquares(factors=100, use_native=True, use_cg=False, regularization=0,
                                    iterations=25)
    model.fit_callback = store_loss(model, 'cholesky')
    model.fit(plays)

    return output


def benchmark_times(plays, iterations=3):
    times = defaultdict(lambda: defaultdict(list))

    def store_time(model, name):
        def inner(iteration, elapsed):
            print(name, model.factors, iteration, elapsed)
            times[name][model.factors].append(elapsed)
        return inner

    output = defaultdict(list)
    for factors in range(32, 257, 32):
        for steps in [2, 3, 4]:
            model = AlternatingLeastSquares(factors=factors, use_native=True, use_cg=True,
                                            regularization=0, iterations=iterations)
            model.fit_callback = store_time(model, 'cg%i' % steps)
            model.cg_steps = steps
            model.fit(plays)

        model = AlternatingLeastSquares(factors=factors, use_native=True, use_cg=False,
                                        regularization=0, iterations=iterations)
        model.fit_callback = store_time(model, 'cholesky')
        model.fit(plays)

        if has_cuda:
            model = AlternatingLeastSquares(factors=factors, use_native=True, use_gpu=True,
                                            regularization=0, iterations=iterations)
            model.fit_callback = store_time(model, 'gpu')
            model.fit(plays)

        # take the min time for the output
        output['factors'].append(factors)
        for name, stats in times.items():
            output[name].append(min(stats[factors]))

    return output


LABELS = {'cg2': 'CG (2 Steps/Iteration)',
          'cg3': 'CG (3 Steps/Iteration)',
          'cg4': 'CG (4 Steps/Iteration)',
          'gpu': 'GPU',
          'cholesky': 'Cholesky'}

COLOURS = {'cg2': "#2ca02c",
           'cg3': "#ff7f0e",
           'cg4': "#c5b0d5",
           'gpu': "#1f77b4",
           'cholesky': "#d62728"}


def generate_speed_graph(data, filename="als_speed.png", keys=['gpu', 'cg2', 'cg3', 'cholesky'],
                         labels=None, colours=None):
    labels = labels or {}
    colours = colours or {}

    seaborn.set()
    fig, ax = plt.subplots()

    factors = data['factors']
    for key in keys:
        ax.plot(factors, data[key],
                color=colours.get(key, COLOURS.get(key)),
                marker='o', markersize=6)

        ax.text(factors[-1] + 5, data[key][-1], labels.get(key, LABELS[key]), fontsize=10)

    ax.set_ylabel("Seconds per Iteration")
    ax.set_xlabel("Factors")
    plt.savefig(filename, bbox_inches='tight', dpi=300)


def generate_loss_graph(data, filename="als_speed.png", keys=['gpu', 'cg2', 'cg3', 'cholesky']):
    seaborn.set()

    fig, ax = plt.subplots()

    iterations = range(1, len(data['cholesky']) + 1)
    for key in keys:
        ax.plot(iterations, data[key],
                color=COLOURS[key],
                marker='o', markersize=6)
        ax.text(iterations[-1] + 1, data[key][-1], LABELS[key], fontsize=10)

    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Iteration")
    plt.savefig(filename, bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CG version against Cholesky",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str, required=True,
                        dest='inputfile', help='dataset file in matrix market format')
    parser.add_argument('--graph', help='generates graphs',
                        action="store_true")
    parser.add_argument('--loss', help='test training loss',
                        action="store_true")
    parser.add_argument('--speed', help='test training speed',
                        action="store_true")

    args = parser.parse_args()
    if not (args.speed or args.loss):
        print("must specify at least one of --speed or --loss")
        parser.print_help()

    else:
        plays = bm25_weight(scipy.io.mmread(args.inputfile)).tocsr()
        logging.basicConfig(level=logging.DEBUG)

        if args.loss:
            acc = benchmark_accuracy(plays)
            json.dump(acc, open("als_accuracy.json", "w"))
            if args.graph:
                generate_loss_graph(acc, "als_accuracy.png")

        if args.speed:
            speed = benchmark_times(plays)
            json.dump(speed, open("als_speed.json", "w"))
            if args.graph:
                generate_speed_graph(speed, "als_speed.png")
