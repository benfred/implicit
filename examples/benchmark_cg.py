""" test script to verify the CG method works, and time it versus cholesky """

from __future__ import print_function

import argparse
import functools
import json
import logging
import time
from collections import defaultdict

import numpy

from implicit._als import calculate_loss, least_squares, least_squares_cg
from implicit.nearest_neighbours import bm25_weight
from lastfm import read_data


def benchmark_solver(Cui, factors, solver, callback, iterations=7, dtype=numpy.float64,
                     regularization=0.00, num_threads=0):
    users, items = Cui.shape

    # have to explode out most of the alternating_least_squares call here
    X = numpy.random.rand(users, factors).astype(dtype) * 0.01
    Y = numpy.random.rand(items, factors).astype(dtype) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    for iteration in range(iterations):
        s = time.time()
        solver(Cui, X, Y, regularization, num_threads=num_threads)
        solver(Ciu, Y, X, regularization, num_threads=num_threads)
        callback(time.time() - s, X, Y)
        logging.debug("finished iteration %i in %s", iteration, time.time() - s)

    return X, Y


def benchmark_accuracy(plays):
    output = defaultdict(list)
    benchmark_solver(plays, 100,
                     least_squares,
                     lambda _, X, Y: output['cholesky'].append(calculate_loss(plays, X, Y,
                                                                              0)),
                     iterations=25)

    for steps in [2, 3, 4]:
        benchmark_solver(plays, 100, functools.partial(least_squares_cg, cg_steps=steps),
                         lambda _, X, Y: output['cg%i' % steps].append(calculate_loss(plays, X, Y,
                                                                                      0)),
                         iterations=25)

    return output


def benchmark_times(plays):
    output = defaultdict(list)
    for factors in [50, 100, 150, 200, 250]:
        output['factors'].append(factors)
        for steps in [2, 3, 4]:
            current = []
            benchmark_solver(plays, factors,
                             functools.partial(least_squares_cg, cg_steps=steps),
                             lambda elapsed, X, Y: current.append(elapsed),
                             iterations=3)
            print("cg%i: %i factors : %ss" % (steps, factors, min(current)))
            output['cg%i' % steps].append(min(current))

        current = []
        benchmark_solver(plays, factors, least_squares,
                         lambda elapsed, X, Y: current.append(elapsed),
                         iterations=3)
        output['cholesky'].append(min(current))
        print("cholesky: %i factors : %ss" % (factors, min(current)))

    return output


def generate_speed_graph(data, filename="cg_training_speed.html"):
    from bokeh.plotting import figure, save
    p = figure(title="Training Time", x_axis_label='Factors', y_axis_label='Seconds')

    to_plot = [(data['cg2'], "CG (2 Steps/Iteration)", "#2ca02c"),
               (data['cg3'], "CG (3 Steps/Iteration)", "#ff7f0e"),
               # (data['cg4'], "CG (4 Steps/Iteration)", "#d62728"),
               (data['cholesky'], "Cholesky", "#1f77b4")]

    p = figure(title="Training Speed", x_axis_label='Factors', y_axis_label='Time / Iteration (s)')
    for current, label, colour in to_plot:
        p.line(data['factors'], current, legend=label, line_color=colour, line_width=1)
        p.circle(data['factors'], current, legend=label, line_color=colour, size=6,
                 fill_color="white")
    p.legend.location = "top_left"
    save(p, filename, title="CG ALS Training Speed")


def generate_loss_graph(data, filename):
    from bokeh.plotting import figure, save

    iterations = range(1, len(data['cholesky']) + 1)
    to_plot = [(data['cg2'], "CG (2 Steps/Iteration)", "#2ca02c"),
               (data['cg3'], "CG (3 Steps/Iteration)", "#ff7f0e"),
               # (data['cg4'], "CG (4 Steps/Iteration)", "#d62728"),
               (data['cholesky'], "Cholesky", "#1f77b4")]

    p = figure(title="Training Loss", x_axis_label='Iteration', y_axis_label='MSE')
    for loss, label, colour in to_plot:
        p.line(iterations, loss, legend=label, line_color=colour, line_width=1)
        p.circle(iterations, loss, legend=label, line_color=colour, size=6, fill_color="white")

    save(p, filename, title="CG ALS Training Loss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CG version against Cholesky",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str,
                        dest='inputfile', help='last.fm dataset file', required=True)
    parser.add_argument('--graph', help='generates graphs (requires bokeh)',
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

        plays = bm25_weight(read_data(args.inputfile)[1]).tocsr()
        logging.basicConfig(level=logging.DEBUG)

        if args.loss:
            acc = benchmark_accuracy(plays)
            json.dump(acc, open("cg_accuracy.json", "w"))
            if args.graph:
                generate_loss_graph(acc, "cg_accuracy.html")

        if args.speed:
            speed = benchmark_times(plays)
            json.dump(speed, open("cg_speed.json", "w"))
            if args.graph:
                generate_speed_graph(speed, "cg_speed.html")
