import itertools
import logging

import numpy
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import augment_inner_product_matrix

log = logging.getLogger("implicit")


class AnnoyALSWrapper:
    """A wrapper of the :class:`~implicit.als.AlternatingLeastSquares` that uses an
    `Annoy <https://github.com/spotify/annoy>`_ index to calculate similar items and
    recommend items.

    Parameters
    ----------
    model: AlternatingLeastSquares, required
        the AlternatingLeastSquares to wrap
    n_trees : int, optional
        The number of trees to use when building the Annoy index. More trees gives higher precision
        when querying.
    search_k : int, optional
        Provides a way to search more trees at runtime, giving the ability to have more accurate
        results at the cost of taking more time.
    approximate_similar_items : bool, optional
        whether or not to build an Annoy index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an Annoy index for the recommend call

    Attributes
    ----------
    similar_items_index : annoy.AnnoyIndex
        Annoy index for looking up similar items in the cosine space formed by the latent
        item_factors

    recommend_index : annoy.AnnoyIndex
        Annoy index for looking up similar items in the inner product space formed by the latent
        item_factors
    """

    def __init__(self, model: AlternatingLeastSquares, approximate_similar_items=True, approximate_recommend=True,
                 n_trees=50, search_k=-1):
        import annoy # delay import in case the library is not installed
        self.model = model

        self.similar_items_index = None
        self.recommend_index = None

        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

        self.n_trees = n_trees
        self.search_k = search_k

        self.max_norm = numpy.nan

    def item_factors(self):
        return self.model.item_factors

    def user_factors(self):
        return self.model.user_factors

    def fit(self, Ciu, show_progress=True):
        self.model.fit(Ciu, show_progress)
        self.initialize()

    def initialize(self):
        # build up an Annoy Index with all the item_factors (for calculating
        # similar items)
        if self.approximate_similar_items:
            log.info("Building annoy similar items index")

            self.similar_items_index = annoy.AnnoyIndex(
                self.model.item_factors.shape[1], 'angular')
            for i, row in enumerate(self.model.item_factors):
                self.similar_items_index.add_item(i, row)
            self.similar_items_index.build(self.n_trees)

        # build up a separate index for the inner product (for recommend
        # methods)
        if self.approximate_recommend:
            log.info("Building annoy recommendation index")
            self.max_norm, extra = augment_inner_product_matrix(self.model.item_factors)
            self.recommend_index = annoy.AnnoyIndex(extra.shape[1], 'angular')
            for i, row in enumerate(extra):
                self.recommend_index.add_item(i, row)
            self.recommend_index.build(self.n_trees)

    def similar_items(self, itemid, N=10):
        if not self.approximate_similar_items:
            return self.model.similar_items(itemid, N)

        neighbours, dist = self.similar_items_index.get_nns_by_item(itemid, N,
                                                                    search_k=self.search_k,
                                                                    include_distances=True)
        # transform distances back to cosine from euclidean distance
        return zip(neighbours, 1 - (numpy.array(dist) ** 2) / 2)

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
        ids, dist = self.recommend_index.get_nns_by_vector(query, count, include_distances=True,
                                                           search_k=self.search_k)

        # convert the distances from euclidean to cosine distance,
        # and then rescale the cosine distance to go back to inner product
        scaling = self.max_norm * numpy.linalg.norm(query)
        dist = scaling * (1 - (numpy.array(dist) ** 2) / 2)
        return list(itertools.islice((rec for rec in zip(ids, dist) if rec[0] not in item_filter), N))
