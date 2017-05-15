from __future__ import print_function

import unittest

from implicit.annoy_als import AnnoyAlternatingLeastSquares, has_annoy

from .recommender_base_test import TestRecommenderBaseMixin

if has_annoy:
    # Annoyingly, 'annoy' doesn't seem to build on windows
    # don't bother testing with this
    class AnnoyALSTest(unittest.TestCase, TestRecommenderBaseMixin):
        def _get_model(self):
            return AnnoyAlternatingLeastSquares(factors=3, regularization=0)

if __name__ == "__main__":
    unittest.main()
