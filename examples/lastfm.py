""" An example of using this library to calculate related artists
from the last.fm dataset. More details can be found
at http://www.benfrederickson.com/matrix_factorization/
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


def bm25_weight(data, K1=100, B=0.8):
    """ Weighs each row of the matrix data by BM25 weighting """
    # calculate idf per term (user)
    N = float(data.shape[0])
    idf = numpy.log(N / (1 + numpy.bincount(data.col)))

    # calculate length_norm per document (artist)
    row_sums = numpy.squeeze(numpy.asarray(data.sum(1)))
    average_length = row_sums.sum() / N
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    ret = coo_matrix(data)
    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
    return ret


class ExactTopRelated(object):
    def __init__(self, artist_factors):
        # fully normalize artist_factors, so can compare
        # with only dot product
        for row in artist_factors:
            row /= (1e-10 + numpy.linalg.norm(row))
        self.artist_factors

    def get_related(self, artistid, N=10):
        scores = self.artist_factors.dot(self.artist_factors[artistid])
        best = numpy.argpartition(scores, -N)[-N:]
        return sorted((best, scores[best]), key=lambda x: -x[1])


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
                              use_native=True):
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
                                                             use_native=use_native)
    logging.debug("calculated factors in %s", time.time() - start)

    # write out artists by popularity
    logging.debug("calculating top artists")
    user_count = df.groupby('artist').size()
    artists = dict(enumerate(df['artist'].cat.categories))
    to_generate = sorted(list(artists), key=lambda x: -user_count[x])

    if exact:
        calc = ExactTopRelated(artist_factors)
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    calculate_similar_artists(args.inputfile, args.outputfile,
                              factors=args.factors,
                              regularization=args.regularization,
                              exact=args.exact, trees=args.treecount,
                              iterations=args.iterations,
                              use_native=not args.purepython)
