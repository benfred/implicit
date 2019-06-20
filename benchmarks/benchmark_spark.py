# Compile spark with native blas support:
# https://github.com/Mega-DatA-Lab/SpectralLDA-Spark/wiki/Compile-Spark-with-Native-BLAS-LAPACK-Support
from __future__ import print_function

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy
import scipy.io
import seaborn
from pyspark import SparkConf, SparkContext
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession

import implicit


def convert_sparse_to_dataframe(spark, context, sparse_matrix):
    """ Converts a scipy sparse matrix to a spark dataframe """
    m = sparse_matrix.tocoo()
    data = context.parallelize(numpy.array([m.row, m.col, m.data]).T,
                               numSlices=len(m.row)/1024)
    return spark.createDataFrame(data.map(lambda p: Row(row=int(p[0]),
                                                        col=int(p[1]),
                                                        data=float(p[2]))))


def benchmark_spark(ratings, factors, iterations=5):
    conf = (SparkConf()
            .setAppName("implicit_benchmark")
            .setMaster('local[*]')
            .set('spark.driver.memory', '16G')
            )
    context = SparkContext(conf=conf)
    spark = SparkSession(context)

    times = {}
    try:
        ratings = convert_sparse_to_dataframe(spark, context, ratings)

        for rank in factors:
            als = ALS(rank=rank, maxIter=iterations,
                      alpha=1, implicitPrefs=True,
                      userCol="row", itemCol="col", ratingCol="data")
            start = time.time()
            als.fit(ratings)
            elapsed = time.time() - start
            times[rank] = elapsed / iterations
            print("spark. factors=%i took %.3f" % (rank, elapsed/iterations))
    finally:
        spark.stop()

    return times


def benchmark_implicit(ratings, factors, iterations=5, use_gpu=False):
    ratings = ratings.tocsr()
    times = {}
    for rank in factors:
        model = implicit.als.AlternatingLeastSquares(factors=rank,
                                                     iterations=iterations,
                                                     use_gpu=use_gpu)
        start = time.time()
        model.fit(ratings)
        elapsed = time.time() - start
        # take average time over iterations to be consistent with spark timings
        times[rank] = elapsed / iterations
        print("implicit. factors=%i took %.3f" % (rank, elapsed/iterations))
    return times


def generate_graph(times, factors, filename="spark_speed.png"):
    seaborn.set()
    fig, ax = plt.subplots()
    for key in times:
        current = [times[key][f] for f in factors]
        ax.plot(factors, current, marker='o', markersize=6)
        ax.text(factors[-1] + 5, current[-1], key, fontsize=10)

    ax.set_ylabel("Seconds per Iteration")
    ax.set_xlabel("Factors")
    plt.savefig(filename, bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Spark against implicit",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str, required=True,
                        help='dataset file in matrix market format')
    parser.add_argument('--output', type=str, required=True,
                        help='output file location')
    args = parser.parse_args()
    if not (args.speed or args.loss):
        print("must specify at least one of --speed or --loss")
        parser.print_help()

    m = scipy.io.mmread(args.inputfile)

    times = {}
    factors = list(range(64, 257, 64))

    times['Implicit (GPU)'] = benchmark_implicit(m, factors, use_gpu=True)
    times['Spark MLlib'] = benchmark_spark(m, factors)
    times['Implicit (CPU)'] = benchmark_implicit(m, factors, use_gpu=False)

    print(times)
    generate_graph(times, factors, filename=args.output + ".png")

    json.dump(times, open(args.output + ".json", "w"))
