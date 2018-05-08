import h5py
import os
import logging
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

from implicit.datasets import _download


log = logging.getLogger("implicit")


URL = 'https://github.com/benfred/recommender_data/releases/download/v1.0/movielens_20m.hdf5'


def get_movielens():
    """ Returns the lastfm360k dataset, downloading locally if necessary.
    Returns a tuple of (artistids, userids, plays) where plays is a COO matrix """

    filename = os.path.join(_download.LOCAL_CACHE_DIR, "movielens_20m.hdf5")
    if not os.path.isfile(filename):
        log.info("Downloading dataset to '%s'", filename)
        _download.download_file(URL, filename)
    else:
        log.info("Using cached dataset at '%s'", filename)

    with h5py.File(filename, 'r') as f:
        m = f.get('movie_user_ratings')
        plays = csr_matrix((m.get('data'), m.get('indices'), m.get('indptr')))
        return np.array(f['movie']), plays


def generate_dataset(path, outputfilename):
    """ Generates a hdf5 movielens datasetfile from the raw datafiles found at:
    https://grouplens.org/datasets/movielens/20m/

    You shouldn't have to run this yourself, and can instead just download the
    output using the 'get_movielens' funciton./
    """
    ratings, movies = _read_dataframes_20M(path)
    _hfd5_from_dataframe(ratings, movies, outputfilename)


def _read_dataframes_20M(path):
    """ reads in the movielens 20M"""
    import pandas

    ratings = pandas.read_csv(os.path.join(path, "ratings.csv"))
    movies = pandas.read_csv(os.path.join(path, "movies.csv"))

    return ratings, movies


def _hfd5_from_dataframe(ratings, movies, outputfilename):
    # transform ratings dataframe into a sparse matrix
    m = coo_matrix((ratings['rating'].astype(np.float32),
                   (ratings['movieId'], ratings['userId']))).tocsr()

    with h5py.File(outputfilename, "w") as f:
        # write out the ratings matrix
        g = f.create_group('movie_user_ratings')
        g.create_dataset("data", data=m.data)
        g.create_dataset("indptr", data=m.indptr)
        g.create_dataset("indices", data=m.indices)

        # write out the titles as a numpy array
        titles = np.empty(shape=(movies.movieId.max()+1,), dtype=np.object)
        titles[movies.movieId] = movies.title
        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset('movie', (len(titles),), dtype=dt)
        dset[:] = titles
