""" An example of using this library to calculate related artists
from the last.fm dataset. More details can be found
at http://www.benfrederickson.com/matrix-factorization/

The dataset here can be found at
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html

Note there are some invalid entries in this dataset, running
this function will clean it up so pandas can read it:
https://github.com/benfred/bens-blog-code/blob/master/distance-metrics/musicdata.py#L39
"""

from __future__ import print_function

import argparse
import logging
import time

import numpy
import pandas
from scipy.sparse import coo_matrix

from implicit.als import AlternatingLeastSquares
from implicit.annoy_als import AnnoyAlternatingLeastSquares
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)


def read_data(filename):
    """ Reads in the last.fm dataset, and returns a tuple of a pandas dataframe
    and a sparse matrix of artist/user/playcount """
    # read in triples of user/artist/playcount from the input dataset
    data = pandas.read_table(filename,
                             usecols=[0, 2, 3],
                             names=['user', 'artist', 'plays'])

    # map each artist and user to a unique numeric value
    data['user'] = data['user'].astype("category")
    data['artist'] = data['artist'].astype("category")

    # create a sparse matrix of all the users/plays
    plays = coo_matrix((data['plays'].astype(float),
                       (data['artist'].cat.codes.copy(),
                        data['user'].cat.codes.copy())))

    return data, plays


def calculate_similar_artists(input_filename, output_filename,
                              model_name="als",
                              factors=50, regularization=0.01,
                              iterations=15,
                              exact=False,
                              use_native=True,
                              dtype=numpy.float64,
                              cg=False):
    logging.debug("Calculating similar artists. This might take a while")

    # read in the input data file
    logging.debug("reading data from %s", input_filename)
    start = time.time()
    df, plays = read_data(input_filename)
    logging.debug("read data file in %s", time.time() - start)

    # generate a recommender model based off the input params
    if model_name == "als":
        if exact:
            model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                            use_native=use_native, use_cg=cg,
                                            dtype=dtype, iterations=iterations)
        else:
            model = AnnoyAlternatingLeastSquares(factors=factors, regularization=regularization,
                                                 use_native=use_native, use_cg=cg,
                                                 dtype=dtype, iterations=iterations)

        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

    elif model_name == "tfidf":
        model = TFIDFRecommender()

    elif model_name == "cosine":
        model = CosineRecommender()

    elif model_name == "bm25":
        model = BM25Recommender(K1=100, B=0.5)

    else:
        raise NotImplementedError("TODO: model %s" % model_name)

    # train the model
    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(plays)
    logging.debug("trained model '%s' in %s", model_name, time.time() - start)

    # write out similar artists by popularity
    logging.debug("calculating top artists")
    user_count = df.groupby('artist').size()
    artists = dict(enumerate(df['artist'].cat.categories))
    to_generate = sorted(list(artists), key=lambda x: -user_count[x])

    # write out as a TSV of artistid, otherartistid, score
    with open(output_filename, "w") as o:
        for artistid in to_generate:
            artist = artists[artistid]
            for other, score in model.similar_items(artistid, 11):
                o.write("%s\t%s\t%s\n" % (artist, artists[other], score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates related artists on the last.fm dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str,
                        dest='inputfile', help='last.fm dataset file', required=True)
    parser.add_argument('--output', type=str, default='similar-artists.tsv',
                        dest='outputfile', help='output file name')
    parser.add_argument('--model', type=str, default='als',
                        dest='model', help='model to calculate (als/bm25/tfidf/cosine)')
    parser.add_argument('--factors', type=int, default=50, dest='factors',
                        help='Number of factors to calculate')
    parser.add_argument('--reg', type=float, default=0.8, dest='regularization',
                        help='regularization weight')
    parser.add_argument('--iter', type=int, default=15, dest='iterations',
                        help='Number of ALS iterations')
    parser.add_argument('--exact', help='compute exact distances (slow)', action="store_true")
    parser.add_argument('--purepython',
                        help='dont use cython extension (slow)',
                        action="store_true")
    parser.add_argument('--float32',
                        help='use 32 bit floating point numbers',
                        action="store_true")
    parser.add_argument('--cg',
                        help='use CG optimizer',
                        action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    calculate_similar_artists(args.inputfile, args.outputfile,
                              model_name=args.model,
                              factors=args.factors,
                              regularization=args.regularization,
                              exact=args.exact,
                              iterations=args.iterations,
                              use_native=not args.purepython,
                              dtype=numpy.float32 if args.float32 else numpy.float64,
                              cg=args.cg)
