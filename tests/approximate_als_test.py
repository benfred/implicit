from __future__ import print_function

import unittest

from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)

from .recommender_base_test import TestRecommenderBaseMixin

# don't require annoy/faiss/nmslib to be installed
try:
    import annoy  # noqa

    class AnnoyALSTest(unittest.TestCase, TestRecommenderBaseMixin):
        def _get_model(self):
            return AnnoyAlternatingLeastSquares(factors=2, regularization=0)
except ImportError:
    pass

try:
    import nmslib  # noqa

    class NMSLibALSTest(unittest.TestCase, TestRecommenderBaseMixin):
        def _get_model(self):
            return NMSLibAlternatingLeastSquares(factors=2, regularization=0,
                                                 index_params={'post': 2})
except ImportError:
    pass

try:
    import faiss  # noqa

    class FaissALSTest(unittest.TestCase, TestRecommenderBaseMixin):
        def _get_model(self):
            return FaissAlternatingLeastSquares(nlist=1, nprobe=1, factors=2, regularization=0)
except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
