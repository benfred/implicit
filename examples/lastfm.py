""" An example of using this library to calculate related artists
from the last.fm dataset. More details can be found
at http://www.benfrederickson.com/matrix-factorization/

This code will automatically download a HDF5 version of the dataset from
GitHub when it is first run. The original dataset can also be found at
http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html
"""
import argparse
import codecs
import logging
import time

import numpy as np
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.lastfm import get_lastfm
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

# maps command line model argument to class name
MODELS = {
    "als": AlternatingLeastSquares,
    "nmslib_als": NMSLibAlternatingLeastSquares,
    "annoy_als": AnnoyAlternatingLeastSquares,
    "faiss_als": FaissAlternatingLeastSquares,
    "tfidf": TFIDFRecommender,
    "cosine": CosineRecommender,
    "bpr": BayesianPersonalizedRanking,
    "lmf": LogisticMatrixFactorization,
    "bm25": BM25Recommender,
}


def get_model(model_name):
    print(f"getting model {model_name}")
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError(f"Unknown Model '{model_name}'")

    # some default params
    if model_name.endswith("als"):
        params = {"factors": 128, "dtype": np.float32}
    elif model_name == "bm25":
        params = {"K1": 100, "B": 0.5}
    elif model_name == "bpr":
        params = {"factors": 63}
    elif model_name == "lmf":
        params = {"factors": 30, "iterations": 40, "regularization": 1.5}
    else:
        params = {}

    return model_class(**params)


def calculate_similar_artists(output_filename, model_name="als"):
    """generates a list of similar artists in lastfm by utilizing the 'similar_items'
    api of the models"""
    artists, _, plays = get_lastfm()

    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if model_name.endswith("als"):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_recommend = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()
    user_plays = plays.T.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(user_plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    # write out similar artists by popularity
    start = time.time()
    logging.debug("calculating top artists")

    user_count = np.ediff1d(plays.indptr)
    to_generate = sorted(np.arange(len(artists)), key=lambda x: -user_count[x])

    # write out as a TSV of artistid, otherartistid, score
    logging.debug("writing similar items")
    with tqdm.tqdm(total=len(to_generate)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            batch_size = 1000
            for startidx in range(0, len(to_generate), batch_size):
                batch = to_generate[startidx : startidx + batch_size]
                ids, scores = model.similar_items(batch, 11)
                for i, artistid in enumerate(batch):
                    artist = artists[artistid]
                    for other, score in zip(ids[i], scores[i]):
                        o.write(f"{artist}\t{artists[other]}\t{score}\n")
                progress.update(batch_size)

    logging.debug("generated similar artists in %0.2fs", time.time() - start)


def calculate_recommendations(output_filename, model_name="als"):
    """Generates artist recommendations for each user in the dataset"""
    # train the model based off input params
    artists, users, plays = get_lastfm()

    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if model_name.endswith("als"):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_similar_items = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()
    user_plays = plays.T.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(user_plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    # generate recommendations for each user and write out to a file
    start = time.time()
    with tqdm.tqdm(total=len(users)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            batch_size = 1000
            to_generate = np.arange(len(users))
            for startidx in range(0, len(to_generate), batch_size):
                batch = to_generate[startidx : startidx + batch_size]
                ids, scores = model.recommend(
                    batch, user_plays[batch], filter_already_liked_items=True
                )
                for i, userid in enumerate(batch):
                    username = users[userid]
                    for other, score in zip(ids[i], scores[i]):
                        o.write(f"{username}\t{artists[other]}\t{score}\n")
                progress.update(batch_size)
    logging.debug("generated recommendations in %0.2fs", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates similar artists on the last.fm dataset"
        " or generates personalized recommendations for each user",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similar-artists.tsv",
        dest="outputfile",
        help="output file name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="als",
        dest="model",
        help=f"model to calculate ({'/'.join(MODELS.keys())})",
    )
    parser.add_argument(
        "--recommend",
        help="Recommend items for each user rather than calculate similar_items",
        action="store_true",
    )
    parser.add_argument(
        "--param", action="append", help="Parameters to pass to the model, formatted as 'KEY=VALUE"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.recommend:
        calculate_recommendations(args.outputfile, model_name=args.model)
    else:
        calculate_similar_artists(args.outputfile, model_name=args.model)
