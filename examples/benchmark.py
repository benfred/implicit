""" A simple benchmark on the last.fm dataset

Compares the running time of this package vs the QMF library from Quora.

On my laptop (2015 Macbook Pro , Dual Core 3.1 GHz Intel Core i7) running
with 50 factors for 15 iterations this is the output:

    QMF finished in 547.933080912
    Implicit finished in 302.997884989
    Implicit is 1.80837262587 times faster

(implicit-mf package was run separately, I estimate it at over 60,000 times
slower on the last.fm dataset - with an estimated running time of around 250 days)
"""
from __future__ import print_function

import logging
import argparse
import time
from subprocess import call

from implicit import alternating_least_squares
from lastfm import read_data, bm25_weight


def benchmark_implicit(matrix, factors, reg, iterations):
    start = time.time()
    alternating_least_squares(matrix, factors, reg, iterations)
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
    plays = bm25_weight(read_data(args.inputfile)[1])

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
                        dest='inputfile', help='last.fm dataset file')
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
