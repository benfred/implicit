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

import logging
import argparse
import time

import numpy
import pandas
from scipy.sparse import coo_matrix
import annoy

from implicit import alternating_least_squares


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


def bm25_weight(X, K1=100, B=0.8):
    """ Weighs each row of the sparse matrix of the data by BM25 weighting """
    # calculate idf per term (user)
    X = coo_matrix(X)
    N = X.shape[0]
    idf = numpy.log(float(N) / (1 + numpy.bincount(X.col)))

    # calculate length_norm per document (artist)
    row_sums = numpy.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X


class TopRelated(object):
    def __init__(self, artist_factors):
        # fully normalize artist_factors, so can compare with only the dot product
        norms = numpy.linalg.norm(artist_factors, axis=-1)
        self.factors = artist_factors / norms[:, numpy.newaxis]

    def get_related(self, artistid, N=10):
        scores = self.factors.dot(self.factors[artistid])
        best = numpy.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])


class ApproximateTopRelated(object):
    def __init__(self, artist_factors, treecount=20):
        index = annoy.AnnoyIndex(artist_factors.shape[1], 'angular')
        for i, row in enumerate(artist_factors):
            index.add_item(i, row)
        index.build(treecount)
        self.index = index

    def get_related(self, artistid, N=10):
        neighbours = self.index.get_nns_by_item(artistid, N)
        return sorted(((other, 1 - self.index.get_distance(artistid, other))
                      for other in neighbours), key=lambda x: -x[1])


def calculate_similar_artists(input_filename, output_filename,
                              factors=50, regularization=0.01,
                              iterations=15,
                              exact=False, trees=20,
                              use_native=True,
                              dtype=numpy.float64):
    logging.debug("Calculating similar artists. This might take a while")
    logging.debug("reading data from %s", input_filename)
    start = time.time()
    df, plays = read_data(input_filename)
    logging.debug("read data file in %s", time.time() - start)

    logging.debug("weighting matrix by bm25")
    weighted = bm25_weight(plays)

    logging.debug("calculating factors")
    start = time.time()
    artist_factors, user_factors = alternating_least_squares(weighted,
                                                             factors=factors,
                                                             regularization=regularization,
                                                             iterations=iterations,
                                                             use_native=use_native,
                                                             dtype=dtype)
    logging.debug("calculated factors in %s", time.time() - start)

    # write out artists by popularity
    logging.debug("calculating top artists")
    user_count = df.groupby('artist').size()
    artists = dict(enumerate(df['artist'].cat.categories))
    to_generate = sorted(list(artists), key=lambda x: -user_count[x])

    if exact:
        calc = TopRelated(artist_factors)
    else:
        calc = ApproximateTopRelated(artist_factors, trees)

    logging.debug("writing top related to %s", output_filename)
    with open(output_filename, "w") as o:
        for artistid in to_generate:
            artist = artists[artistid]
            for other, score in calc.get_related(artistid):
                o.write("%s\t%s\t%s\n" % (artist, artists[other], score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates related artists on the last.fm dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str,
                        dest='inputfile', help='last.fm dataset file', required=True)
    parser.add_argument('--output', type=str, default='similar-artists.tsv',
                        dest='outputfile', help='output file name')
    parser.add_argument('--factors', type=int, default=50, dest='factors',
                        help='Number of factors to calculate')
    parser.add_argument('--reg', type=float, default=0.8, dest='regularization',
                        help='regularization weight')
    parser.add_argument('--iter', type=int, default=15, dest='iterations',
                        help='Number of ALS iterations')
    parser.add_argument('--exact', help='compute exact distances (slow)', action="store_true")
    parser.add_argument('--trees', type=int, default=20, dest='treecount',
                        help='Number of trees to use in annoy')
    parser.add_argument('--purepython',
                        help='dont use cython extension (slow)',
                        action="store_true")
    parser.add_argument('--float32',
                        help='use 32 bit floating point numbers',
                        action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    calculate_similar_artists(args.inputfile, args.outputfile,
                              factors=args.factors,
                              regularization=args.regularization,
                              exact=args.exact, trees=args.treecount,
                              iterations=args.iterations,
                              use_native=not args.purepython,
                              dtype=numpy.float32 if args.float32 else numpy.float64)

