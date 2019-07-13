from __future__ import print_function

import unittest

from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.cuda import HAS_CUDA

from .recommender_base_test import TestRecommenderBaseMixin

# don't require annoy/faiss/nmslib to be installed
try:
    import annoy  # noqa

    class AnnoyALSTest(unittest.TestCase, TestRecommenderBaseMixin):
        def _get_model(self):
            return AnnoyAlternatingLeastSquares(factors=2, regularization=0, use_gpu=False)

        def test_pickle(self):
            # pickle isn't supported on annoy indices
            pass

except ImportError:
    pass

try:
    import nmslib  # noqa

    class NMSLibALSTest(unittest.TestCase, TestRecommenderBaseMixin):
        def _get_model(self):
            return NMSLibAlternatingLeastSquares(factors=2, regularization=0,
                                                 index_params={'post': 2}, use_gpu=False)

        def test_pickle(self):
            # pickle isn't supported on nmslib indices
            pass

except ImportError:
    pass

try:
    import faiss  # noqa

    class FaissALSTest(unittest.TestCase, TestRecommenderBaseMixin):
        def _get_model(self):
            return FaissAlternatingLeastSquares(nlist=1, nprobe=1, factors=2, regularization=0,
                                                use_gpu=False)

        def test_pickle(self):
            # pickle isn't supported on faiss indices
            pass

    if HAS_CUDA:
        class FaissALSGPUTest(unittest.TestCase, TestRecommenderBaseMixin):
            __regularization = 0

            def _get_model(self):
                return FaissAlternatingLeastSquares(nlist=1, nprobe=1, factors=32,
                                                    regularization=self.__regularization,
                                                    use_gpu=True)

            def test_similar_items(self):
                # For the GPU version, we currently have to have factors be a multiple of 32
                # (limitation that I think is caused by how we are currently calculating the
                # dot product in CUDA, TODO: eventually should fix that code).
                # this causes the test_similar_items call to fail if we set regularization to 0
                self.__regularization = 1.0
                try:
                    super(FaissALSGPUTest, self).test_similar_items()
                finally:
                    self.__regularization = 0.0

            def test_large_recommend(self):
                # the GPU version of FAISS can't return more than 1K result (and will assert/exit)
                # this tests out that we fall back in this case to the exact version and don't die
                plays = self.get_checker_board(2048)
                model = self._get_model()
                model.fit(plays, show_progress=False)

                recs = model.similar_items(0, N=1050)
                self.assertEqual(recs[0][0], 0)

                recs = model.recommend(0, plays.T.tocsr(), N=1050)
                self.assertEqual(recs[0][0], 0)

            def test_pickle(self):
                # pickle isn't supported on faiss indices
                pass

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
