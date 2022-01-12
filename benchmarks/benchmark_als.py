""" test script to verify the CG method works, and time it versus cholesky """
import argparse
import json
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import scipy.io
import seaborn

from implicit.als import AlternatingLeastSquares
from implicit.gpu import HAS_CUDA
from implicit.nearest_neighbours import bm25_weight


def benchmark_accuracy(plays):
    output = defaultdict(list)

    def store_loss(name):
        def inner(iteration, elapsed, loss):
            print(f"model {name} iteration {iteration} loss {loss:.5f}")
            output[name].append(loss)

        return inner

    for steps in [2, 3, 4]:
        model = AlternatingLeastSquares(
            factors=128,
            use_gpu=False,
            regularization=0,
            iterations=25,
            calculate_training_loss=True,
        )
        model.cg_steps = steps
        model.fit_callback = store_loss(f"cg{steps}")
        model.fit(plays)

    if HAS_CUDA:
        model = AlternatingLeastSquares(
            factors=128,
            use_native=True,
            use_gpu=True,
            regularization=0,
            iterations=25,
            calculate_training_loss=True,
        )
        model.fit_callback = store_loss("gpu")
        model.use_gpu = True
        model.fit(plays)

    model = AlternatingLeastSquares(
        factors=128,
        use_native=True,
        use_cg=False,
        use_gpu=False,
        regularization=0,
        iterations=25,
        calculate_training_loss=True,
    )
    model.fit_callback = store_loss("cholesky")
    model.fit(plays)

    return output


def benchmark_times(plays, iterations=3):
    times = defaultdict(lambda: defaultdict(list))

    def store_time(model, name):
        def inner(iteration, elapsed, loss):
            print(name, model.factors, iteration, elapsed)
            times[name][model.factors].append(elapsed)

        return inner

    output = defaultdict(list)
    for factors in range(32, 257, 32):
        for steps in [2, 3, 4]:
            model = AlternatingLeastSquares(
                factors=factors,
                use_native=True,
                use_cg=True,
                use_gpu=False,
                regularization=0,
                iterations=iterations,
            )
            model.fit_callback = store_time(model, f"cg{steps}")
            model.cg_steps = steps
            model.fit(plays)

        model = AlternatingLeastSquares(
            factors=factors,
            use_native=True,
            use_cg=False,
            regularization=0,
            iterations=iterations,
            use_gpu=False,
        )
        model.fit_callback = store_time(model, "cholesky")
        model.fit(plays)

        if HAS_CUDA:
            model = AlternatingLeastSquares(
                factors=factors,
                use_native=True,
                use_gpu=True,
                regularization=0,
                iterations=iterations,
            )
            model.fit_callback = store_time(model, "gpu")
            model.fit(plays)

        # take the min time for the output
        output["factors"].append(factors)
        for name, stats in times.items():
            output[name].append(min(stats[factors]))

    return output


LABELS = {
    "cg2": "CG (2 Steps/Iteration)",
    "cg3": "CG (3 Steps/Iteration)",
    "cg4": "CG (4 Steps/Iteration)",
    "gpu": "GPU",
    "cholesky": "Cholesky",
}

COLOURS = {
    "cg2": "#2ca02c",
    "cg3": "#ff7f0e",
    "cg4": "#c5b0d5",
    "gpu": "#1f77b4",
    "cholesky": "#d62728",
}


def generate_speed_graph(
    data,
    filename="als_speed.png",
    labels=None,
    colours=None,
):
    labels = labels or {}
    colours = colours or {}

    seaborn.set()
    _, ax = plt.subplots()

    factors = data["factors"]
    for key in data.keys():
        ax.plot(
            factors, data[key], color=colours.get(key, COLOURS.get(key)), marker="o", markersize=6
        )

        ax.text(factors[-1] + 5, data[key][-1], labels.get(key, LABELS[key]), fontsize=10)

    ax.set_ylabel("Seconds per Iteration")
    ax.set_xlabel("Factors")
    plt.savefig(filename, bbox_inches="tight", dpi=300)


def generate_loss_graph(data, filename="als_speed.png"):
    seaborn.set()

    _, ax = plt.subplots()

    iterations = range(1, len(data["cholesky"]) + 1)
    for key in data.keys():
        ax.plot(iterations, data[key], color=COLOURS[key], marker="o", markersize=6)
        ax.text(iterations[-1] + 1, data[key][-1], LABELS[key], fontsize=10)

    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Iteration")
    plt.savefig(filename, bbox_inches="tight", dpi=300)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CG version against Cholesky",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        dest="inputfile",
        help="dataset file in matrix market format",
    )
    parser.add_argument("--graph", help="generates graphs", action="store_true")
    parser.add_argument("--loss", help="test training loss", action="store_true")
    parser.add_argument("--speed", help="test training speed", action="store_true")

    args = parser.parse_args()
    if not (args.speed or args.loss):
        print("must specify at least one of --speed or --loss")
        parser.print_help()

    else:
        plays = bm25_weight(scipy.io.mmread(args.inputfile)).tocsr()
        logging.basicConfig(level=logging.DEBUG)

        if args.loss:
            acc = benchmark_accuracy(plays)
            with open("als_accuracy.json", "w", encoding="utf8") as o:
                json.dump(acc, o)
            if args.graph:
                generate_loss_graph(acc, "als_accuracy.png")

        if args.speed:
            speed = benchmark_times(plays)
            with open("als_speed.json", "w", encoding="utf8") as o:
                json.dump(speed, o)
            if args.graph:
                generate_speed_graph(speed, "als_speed.png")


if __name__ == "__main__":
    main()
