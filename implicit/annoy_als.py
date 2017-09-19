""" Uses Annoy (https://github.com/spotify/annoy) to quickly retrieve
approximate neighbours from an ALS Matrix factorization model
"""
import itertools
import logging

import numpy
import joblib

from implicit.als import AlternatingLeastSquares

try:
    import annoy
    has_annoy = True
except ImportError:
    has_annoy = False
    logging.warning("Annoy isn't installed")


class MaximumInnerProductIndex(object):
    """ This class uses an Annoy Index to return the top related items by
    the inner product - instead of by the cosine.

    This involves transforming each vector by adding one extra dimension
    as suggested in the paper:
    "Speeding Up the Xbox Recommender System Using a Euclidean Transformation for
    Inner-Product Spaces"
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

    Basically this involves transforming each feature vector so that they have the same
    norm, which means the cosine of this transformed vector is proportional to the
    dot product (if the other vector in the cosine has a 0 in the extra dimension).
    """
    def __init__(self, factors, num_trees=None):
        num_trees = num_trees or factors.shape[1]

        # figure out the norms/ max norm for each row in the matrix
        norms = numpy.linalg.norm(factors, axis=1)
        self.max_norm = norms.max()

        # add an extra dimension so that the norm of each row is the same (max_norm)
        extra_dimension = numpy.sqrt(self.max_norm ** 2 - norms ** 2)
        extra = numpy.append(factors, extra_dimension.reshape(norms.shape[0], 1), axis=1)

        # add the extended matrix to an annoy index
        self.index_dim = factors.shape[1] + 1
        index = annoy.AnnoyIndex(self.index_dim, 'angular')
        for i, row in enumerate(extra):
            index.add_item(i, row)

        index.build(num_trees)
        self.index = index
    
    @classmethod
    def from_file(cls, base_filename):
        """ Factory method to create a MaximumInnerProductIndex model from :filename: """
        obj = joblib.load(base_filename + '.pkl')
        index = annoy.AnnoyIndex(obj.index_dim, 'angular')
        index.load(base_filename + '_index.bin')

        obj.index = index
        return obj

    def to_file(self, base_filename):
        """ Save the current MaximumInnerProductIndex to file."""
        # Make None so that not trying to pickle Annoy object
        index = self.index
        self.index = None

        joblib.dump(self, base_filename + '.pkl', compress=3)
        index.save(base_filename + '_index.bin')

    def get_nns_by_vector(self, v, N=10):
        return self._get_nns(numpy.append(v, 0), N)

    def get_nns_by_item(self, itemid, N=10):
        v = self.index.get_item_vector(itemid)
        v[-1] = 0
        return self._get_nns(v)

    def _get_nns(self, v, N=10):
        ids, dist = self.index.get_nns_by_vector(v, N, include_distances=True)

        # convert the distances from euclidean to cosine distance,
        # and then rescale the cosine distance to go back to inner product
        scaling = self.max_norm * numpy.linalg.norm(v)
        return ids, scaling * (1 - (numpy.array(dist) ** 2) / 2)


class AnnoyAlternatingLeastSquares(AlternatingLeastSquares):
    """ A version of the AlternatingLeastSquares model that uses an annoy
    index to calculate similar items. This leads to massive speedups
    when called repeatedly """
    def fit(self, Ciu):
        # train the model
        super(AnnoyAlternatingLeastSquares, self).fit(Ciu)

        self.index_dim = self.item_factors.shape[1]

        # build up an Annoy Index with all the item_factors (for calculating similar items)
        self.cosine_index = annoy.AnnoyIndex(self.index_dim, 'angular')
        for i, row in enumerate(self.item_factors):
            self.cosine_index.add_item(i, row)
        self.cosine_index.build(self.factors)

        # build up a separate index for the inner product (for recommend methods)
        self.inner_product_index = MaximumInnerProductIndex(self.item_factors)

    def similar_items(self, artistid, N=10):
        neighbours, dist = self.cosine_index.get_nns_by_item(artistid, N, include_distances=True)
        # transform distances back to cosine from euclidean distance
        return zip(neighbours, 1 - (numpy.array(dist) ** 2) / 2)

    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        user = self._user_factor(userid, user_items, recalculate_user)

        # calculate the top N items, removing the users own liked items from the results
        liked = set(user_items[userid].indices)
        if filter_items:
            liked.update(filter_items)
        count = N + len(liked)

        # get the top items by dot product
        ids, dist = self.inner_product_index.get_nns_by_vector(user, count)
        return list(itertools.islice((rec for rec in zip(ids, dist) if rec[0] not in liked), N))

    def to_files(self, base_filename):
        """
        Saves an Annoy ALS to filenames:
        :base_filename: + '_annoy_als.pkl'
        :base_filename: + '_inner_product_index.pkl'
        :base_filename: + '_cosine_index.pkl'

        *** Warning: If future Annoy fields are added in future, will have to
        save / restore these separately as well.
        """
        # Remove fields with annoy.AnnoyIndex members
        inner_product_index = self.inner_product_index
        cosine_index = self.cosine_index

        self.inner_product_index = None
        self.cosine_index = None

        joblib.dump(self, '{}_annoy_als.pkl'.format(base_filename), compress=3)
        inner_product_index.to_file(base_filename + '_inner_product_index')
        cosine_index.save('{}_cosine_index.bin'.format(base_filename))

    

    @classmethod
    def from_files(cls, base_filename):
        """
        Factory to load AnnoyAlternatingLeastSquares model from files.

        :base_filename: describes the base filename that the inner product index
        and cosine index have been saved to.
        """
        obj = joblib.load('{}_annoy_als.pkl'.format(base_filename))
        obj.inner_product_index = MaximumInnerProductIndex.from_file(base_filename + '_inner_product_index')

        obj.cosine_index = annoy.AnnoyIndex(obj.index_dim, 'angular')

        obj.cosine_index.load('{}_cosine_index.bin'.format(base_filename))
        return obj
