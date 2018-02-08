import unittest

from implicit.bpr import BayesianPersonalizedRanking

from .recommender_base_test import TestRecommenderBaseMixin


class BPRTest(unittest.TestCase, TestRecommenderBaseMixin):

    def _get_model(self):
        return BayesianPersonalizedRanking(factors=3, regularization=0)


try:
    import implicit.cuda  # noqa

    class BPRGPUTest(unittest.TestCase, TestRecommenderBaseMixin):

        def _get_model(self):
            return BayesianPersonalizedRanking(factors=3, regularization=0, use_gpu=True)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
