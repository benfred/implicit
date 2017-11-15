""" A simple benchmark comparing the ALS model here to QMF from Quora.

Compares the running time of this package vs the QMF library from Quora.

On my desktop (Intel Core i7 7820x) running with 50 factors for 15 iterations
on the last.fm 360k dataset, this is the output:

    QMF finished in 279.32511353492737
    Implicit finished in 24.046602964401245
    Implicit is 11.615990580808532 times faster
"""
from __future__ import print_function

import argparse
import logging
import time
from subprocess import call

import scipy.io

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight


def benchmark_implicit(matrix, factors, reg, iterations):
    start = time.time()
    model = AlternatingLeastSquares(factors, regularization=reg, iterations=iterations, use_cg=True)
    model.fit(matrix)
    return time.time() - start


def benchmark_qmf(qmfpath, matrix, factors, reg, iterations):
    matrix = matrix.tocoo()
    datafile = "qmf_data.txt"
    open(datafile, "w").write("\n".join("%s %s %s" % vals
                                        for vals in zip(matrix.row, matrix.col, matrix.data)))

    def get_qmf_command(nepochs):
        return [qmfpath, "--train_dataset", datafile,
                "--nfactors", str(factors),
                "--confidence_weight", "1",
                "--nepochs", str(nepochs),
                "--regularization_lambda", str(reg)]

    # ok, so QMF needs to read the data in - and including
    # that in the timing isn't fair. So run it once with no iterations
    # to get a sense of how long reading the input data takes, and
    # subtract from the final results
    read_start = time.time()
    call(get_qmf_command(0))
    read_dataset_time = time.time() - read_start

    calculate_start = time.time()
    call(get_qmf_command(iterations))
    return time.time() - calculate_start - read_dataset_time


def run_benchmark(args):
    plays = bm25_weight(scipy.io.mmread(args.inputfile))

    qmf_time = benchmark_qmf(args.qmfpath, plays, args.factors, args.regularization,
                             args.iterations)

    implicit_time = benchmark_implicit(plays, args.factors, args.regularization, args.iterations)

    print("QMF finished in", qmf_time)
    print("Implicit finished in", implicit_time)
    print("Implicit is %s times faster" % (qmf_time / implicit_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates Benchmark",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str,
                        dest='inputfile', help='dataset file in matrix market format')
    parser.add_argument('--qmfpath', type=str,
                        dest='qmfpath', help='full path to qmf wals.bin file', required=True)
    parser.add_argument('--factors', type=int, default=50, dest='factors',
                        help='Number of factors to calculate')
    parser.add_argument('--reg', type=float, default=0.8, dest='regularization',
                        help='regularization weight')
    parser.add_argument('--iter', type=int, default=15, dest='iterations',
                        help='Number of ALS iterations')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    run_benchmark(args)
