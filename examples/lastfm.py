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
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)

# maps command line model argument to class name
MODELS = {"als":  AlternatingLeastSquares,
          "nmslib_als": NMSLibAlternatingLeastSquares,
          "annoy_als": AnnoyAlternatingLeastSquares,
          "faiss_als": FaissAlternatingLeastSquares,
          "tfidf": TFIDFRecommender,
          "cosine": CosineRecommender,
          "bm25": BM25Recommender}


def get_model(model_name):
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if issubclass(model_class, AlternatingLeastSquares):
        params = {'factors': 64, 'dtype': numpy.float32, 'use_gpu': False}
    elif model_name == "bm25":
        params = {'K1': 100, 'B': 0.5}
    else:
        params = {}

    return model_class(**params)


def read_data(filename):
    """ Reads in the last.fm dataset, and returns a tuple of a pandas dataframe
    and a sparse matrix of artist/user/playcount """
    # read in triples of user/artist/playcount from the input dataset
    # get a model based off the input params
    start = time.time()
    logging.debug("reading data from %s", filename)
    data = pandas.read_table(filename,
                             usecols=[0, 2, 3],
                             names=['user', 'artist', 'plays'])

    # map each artist and user to a unique numeric value
    data['user'] = data['user'].astype("category")
    data['artist'] = data['artist'].astype("category")

    # create a sparse matrix of all the users/plays
    plays = coo_matrix((data['plays'].astype(numpy.float32),
                       (data['artist'].cat.codes.copy(),
                        data['user'].cat.codes.copy())))

    logging.debug("read data file in %s", time.time() - start)
    return data, plays


def calculate_similar_artists(input_filename, output_filename, model_name="als"):
    """ generates a list of similar artists in lastfm by utiliizing the 'similar_items'
    api of the models """
    df, plays = read_data(input_filename)

    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_recommend = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    # write out similar artists by popularity
    artists = dict(enumerate(df['artist'].cat.categories))
    start = time.time()
    logging.debug("calculating top artists")
    user_count = df.groupby('artist').size()
    to_generate = sorted(list(artists), key=lambda x: -user_count[x])

    # write out as a TSV of artistid, otherartistid, score
    with open(output_filename, "w") as o:
        for artistid in to_generate:
            artist = artists[artistid]
            for other, score in model.similar_items(artistid, 11):
                o.write("%s\t%s\t%s\n" % (artist, artists[other], score))

    logging.debug("generated similar artists in %0.2fs",  time.time() - start)


def calculate_recommendations(input_filename, output_filename, model_name="als"):
    """ Generates artist recommendations for each user in the dataset """
    # train the model based off input params
    df, plays = read_data(input_filename)

    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_similar_items = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    # generate recommendations for each user and write out to a file
    artists = dict(enumerate(df['artist'].cat.categories))
    start = time.time()
    user_plays = plays.T.tocsr()
    with open(output_filename, "w") as o:
        for userid, username in enumerate(df['user'].cat.categories):
            for artistid, score in model.recommend(userid, user_plays):
                o.write("%s\t%s\t%s\n" % (username, artists[artistid], score))
    logging.debug("generated recommendations in %0.2fs",  time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates similart artists on the last.fm dataset"
                                     " or generates personalized recommendations for each user",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str,
                        dest='inputfile', help='last.fm dataset file', required=True)
    parser.add_argument('--output', type=str, default='similar-artists.tsv',
                        dest='outputfile', help='output file name')
    parser.add_argument('--model', type=str, default='als',
                        dest='model', help='model to calculate (%s)' % "/".join(MODELS.keys()))
    parser.add_argument('--recommend',
                        help='Recommend items for each user rather than calculate similar_items',
                        action="store_true")
    parser.add_argument('--param', action='append',
                        help="Parameters to pass to the model, formatted as 'KEY=VALUE")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.recommend:
        calculate_recommendations(args.inputfile, args.outputfile, model_name=args.model)
    else:
        calculate_similar_artists(args.inputfile, args.outputfile, model_name=args.model)
