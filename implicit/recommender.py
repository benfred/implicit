import numpy as np
from scipy import sparse

from implicit.als import AlternatingLeastSquares


def read_from_file(file_name: str):
    """
    Reads a model from a '.npz' file
    """
    data = np.load(file_name)
    model = AlternatingLeastSquares(factors=data['model.item_factors'].shape[1])
    model.item_factors = data['model.item_factors']
    model.YtY  # This will initialize the _YtY instance variable which is used directly in internal methods
    if 'user_factors' in data:
        model.user_factors = data['model.user_factors']
    return Recommender(model=model, user_labels=data['user_labels'], item_labels=data['item_labels'])


class RecommenderException(Exception):
    pass


class Recommender:

    def __init__(self, model: AlternatingLeastSquares,
                 user_labels: np.ndarray, item_labels: np.ndarray):
        self.model = model
        self.recommender = model
        self.user_labels = user_labels
        self.item_labels = item_labels
        self.user_labels_idx = {idx: label for label, idx in enumerate(user_labels)}
        self.item_labels_idx = {idx: label for label, idx in enumerate(item_labels)}

    def get_item_label(self, item_id):
        return self.item_labels_idx.get(item_id)

    def get_item_id(self, item_label):
        return self.item_labels[item_label]

    def get_user_label(self, user_id):
        return self.user_labels_idx.get(user_id)

    def get_user_id(self, user_label):
        return self.user_labels[user_label]

    def optimize(self, optimization: str, approximate_similar_items=True, approximate_recommend=True):
        if optimization == 'annoy':
            from implicit.annoy_als import AnnoyALSWrapper
            self.recommender = AnnoyALSWrapper(model=self.model, approximate_similar_items=approximate_similar_items,
                                               approximate_recommend=approximate_recommend)
            self.recommender.initialize()
        elif optimization == 'nmslib':
            from implicit.nmslib_als import NMSLibALSWrapper
            self.recommender = NMSLibALSWrapper(model=self.model, approximate_similar_items=approximate_similar_items,
                                                approximate_recommend=approximate_recommend)
            self.recommender.initialize()
        elif optimization is None:
            self.recommender = self.model
        else:
            raise RecommenderException("Unsupported optimization " + optimization)

    def save(self, file_name, user_factors=False, compress=False):
        data = {
            'model.item_factors': self.model.item_factors,
            'user_labels': self.user_labels,
            'item_labels': self.item_labels,
        }
        if user_factors:
            data.update({'model.user_factors': self.model.user_factors})
        if compress:
            np.savez_compressed(file_name, **data)
        else:
            np.savez(file_name, **data)

    def recommend(self, item_ids, item_weights=None, number_of_results=50, filter_already_liked_items=True):
        """

        :param item_ids:
        :param item_weights:
        :param number_of_results:
        :param filter_already_liked_items:
        :return:
        """
        item_lb = [self.get_item_label(i) for i in item_ids]
        user_ll = [0] * len(item_ids)
        confidence = [10] * len(item_ids) if item_weights is None else item_weights
        user_items = sparse.csr_matrix((confidence, (user_ll, item_lb)))
        user_label = 0

        recommendations = self.recommender.recommend(user_label, user_items=user_items, N=number_of_results,
                                                     recalculate_user=True,
                                                     filter_already_liked_items=filter_already_liked_items)

        recommendations = [(self.get_item_id(x[0]), x[1]) for x in recommendations]

        return recommendations
