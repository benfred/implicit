import itertools
import logging

import numpy
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import augment_inner_product_matrix

log = logging.getLogger("implicit")

logging.getLogger('nmslib').setLevel(logging.WARNING)


class NMSLibALSWrapper:
    """A wrapper of the :class:`~implicit.als.AlternatingLeastSquares` that uses
    `NMSLib <https://github.com/searchivarius/nmslib>`_ to create approximate nearest neighbours
    indices of the latent factors.

    Parameters
    ----------
    model: AlternatingLeastSquares, required
        the AlternatingLeastSquares to wrap
    method : str, optional
        The NMSLib method to use
    index_params: dict, optional
        Optional params to send to the createIndex call in NMSLib
    query_params: dict, optional
        Optional query time params for the NMSLib 'setQueryTimeParams' call
    approximate_similar_items : bool, optional
        whether or not to build an NMSLIB index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an NMSLIB index for the recommend call

    Attributes
    ----------
    similar_items_index : nmslib.FloatIndex
        NMSLib index for looking up similar items in the cosine space formed by the latent
        item_factors

    recommend_index : nmslib.FloatIndex
        NMSLib index for looking up similar items in the inner product space formed by the latent
        item_factors
    """

    def __init__(self, model: AlternatingLeastSquares,
                 approximate_similar_items=True, approximate_recommend=True,
                 method='hnsw', index_params=None, query_params=None):
        import nmslib # delay import in case the library is not installed
        self.model = model
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}
        if query_params is None:
            query_params = {'ef': 90}

        self.similar_items_index = None
        self.recommend_index = None

        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend
        self.method = method

        self.index_params = index_params
        self.query_params = query_params

        self.max_norm = numpy.nan

    def fit(self, Ciu, show_progress=True):
        self.model.fit(Ciu, show_progress)
        self.initialize(show_progress)

    def initialize(self, show_progress=True):
        # create index for similar_items
        if self.approximate_similar_items:
            log.info("Building nmslib similar items index")
            self.similar_items_index = nmslib.init(
                method=self.method, space='cosinesimil')

            # there are some numerical instability issues here with
            # building a cosine index with vectors with 0 norms, hack around this
            # by just not indexing them
            norms = numpy.linalg.norm(self.model.item_factors, axis=1)
            ids = numpy.arange(self.model.item_factors.shape[0])

            # delete zero valued rows from the matrix
            item_factors = numpy.delete(self.model.item_factors, ids[norms == 0], axis=0)
            ids = ids[norms != 0]

            self.similar_items_index.addDataPointBatch(item_factors, ids=ids)
            self.similar_items_index.createIndex(self.index_params,
                                                 print_progress=show_progress)
            self.similar_items_index.setQueryTimeParams(self.query_params)

        # build up a separate index for the inner product (for recommend
        # methods)
        if self.approximate_recommend:
            log.debug("Building nmslib recommendation index")
            self.max_norm, extra = augment_inner_product_matrix(
                self.model.item_factors)
            self.recommend_index = nmslib.init(
                method='hnsw', space='cosinesimil')
            self.recommend_index.addDataPointBatch(extra)
            self.recommend_index.createIndex(self.index_params, print_progress=show_progress)
            self.recommend_index.setQueryTimeParams(self.query_params)

    def similar_items(self, itemid, N=10):
        if not self.approximate_similar_items:
            return self.model.similar_items(itemid, N)

        neighbours, distances = self.similar_items_index.knnQuery(
            self.model.item_factors[itemid], N)
        return zip(neighbours, 1.0 - distances)

    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False,
                  filter_already_liked_items=False):
        if not self.approximate_recommend:
            return self.model.recommend(userid, user_items, N=N,
                                        filter_items=filter_items,
                                        recalculate_user=recalculate_user,
                                        filter_already_liked_items=filter_already_liked_items)

        user = self.model._user_factor(userid, user_items, recalculate_user)

        # calculate the top N items, removing the users own liked items from
        # the results
        item_filter = set(filter_items) if filter_items else set()
        if filter_already_liked_items:
            item_filter.update(user_items[userid].indices)
        count = N + len(item_filter)

        query = numpy.append(user, 0)
        ids, dist = self.recommend_index.knnQuery(query, count)

        # convert the distances from euclidean to cosine distance,
        # and then rescale the cosine distance to go back to inner product
        scaling = self.max_norm * numpy.linalg.norm(query)
        dist = scaling * (1.0 - dist)
        return list(itertools.islice((rec for rec in zip(ids, dist) if rec[0] not in item_filter), N))
