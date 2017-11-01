"""
An example based off the MovieLens 20M dataset, found here
https://grouplens.org/datasets/movielens/

Since this dataset contains explicit 5-star ratings, the ratings are
filtered down to positive reviews (4+ stars) to construct an implicit
dataset
"""

from __future__ import print_function

import argparse
import logging
import os
import time

import numpy
import pandas
from scipy.sparse import coo_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)


def read_data(path, min_rating=4.0):
    """ Reads in the dataset, and filters down ratings down to positive only"""
    ratings = pandas.read_csv(os.path.join(path, "ratings.csv"))
    positive = ratings[ratings.rating >= min_rating]

    movies = pandas.read_csv(os.path.join(path, "movies.csv"))

    m = coo_matrix((positive['rating'].astype(numpy.float32),
                    (positive['movieId'], positive['userId'])))
    m.data = numpy.ones(len(m.data))
    return ratings, movies, m


def calculate_similar_movies(input_path, output_filename,
                             model_name="als", min_rating=4.0):
    # read in the input data file
    logging.debug("reading data from %s", input_path)
    start = time.time()
    ratings, movies, m = read_data(input_path, min_rating=min_rating)
    logging.debug("read data file in %s", time.time() - start)

    # generate a recommender model based off the input params
    if model_name == "als":
        model = AlternatingLeastSquares()

        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        m = bm25_weight(m,  B=0.9) * 5

    elif model_name == "tfidf":
        model = TFIDFRecommender()

    elif model_name == "cosine":
        model = CosineRecommender()

    elif model_name == "bm25":
        model = BM25Recommender(B=0.2)

    else:
        raise NotImplementedError("TODO: model %s" % model_name)

    # train the model
    m = m.tocsr()
    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(m)
    logging.debug("trained model '%s' in %s", model_name, time.time() - start)
    logging.debug("calculating top movies")

    user_count = ratings.groupby('movieId').size()
    movie_lookup = dict((i, m) for i, m in zip(movies['movieId'], movies['title']))
    to_generate = sorted(list(movies['movieId']), key=lambda x: -user_count.get(x, 0))

    with open(output_filename, "w") as o:
        for movieid in to_generate:
            # if this movie has no ratings, skip over (for instance 'Graffiti Bridge' has
            # no ratings > 4 meaning we've filtered out all data for it.
            if m.indptr[movieid] == m.indptr[movieid + 1]:
                continue

            movie = movie_lookup[movieid]
            for other, score in model.similar_items(movieid, 11):
                o.write("%s\t%s\t%s\n" % (movie, movie_lookup[other], score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates related movies from the MovieLens 20M "
                                     "dataset (https://grouplens.org/datasets/movielens/20m/)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str,
                        dest='inputfile', help='Path of the unzipped ml-20m dataset', required=True)
    parser.add_argument('--output', type=str, default='similar-movies.tsv',
                        dest='outputfile', help='output file name')
    parser.add_argument('--model', type=str, default='als',
                        dest='model', help='model to calculate (als/bm25/tfidf/cosine)')
    parser.add_argument('--min_rating', type=float, default=4.0, dest='min_rating',
                        help='Minimum rating to assume that a rating is positive')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    calculate_similar_movies(args.inputfile, args.outputfile,
                             model_name=args.model,
                             min_rating=args.min_rating)
