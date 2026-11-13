import unittest

from recommender_base_test import RecommenderBaseTestMixin, get_checker_board

from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    MLXVSAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.gpu import HAS_CUDA

# don't require annoy/faiss/nmslib to be installed
try:
    import annoy  # noqa pylint: disable=unused-import

    class AnnoyALSTest(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return AnnoyAlternatingLeastSquares(
                factors=32, regularization=0, random_state=23, use_gpu=False
            )

        def test_pickle(self):
            # pickle isn't supported on annoy indices
            pass

        def test_serialization(self):
            pass

        def test_serialization_without_fit(self):
            pass

    if HAS_CUDA:

        class AnnoyALSGPUTest(unittest.TestCase, RecommenderBaseTestMixin):
            def _get_model(self):
                return AnnoyAlternatingLeastSquares(
                    factors=32, regularization=0, random_state=23, use_gpu=True
                )

            def test_pickle(self):
                # pickle isn't supported on annoy indices
                pass

            def test_serialization(self):
                pass

            def test_serialization_without_fit(self):
                pass

except ImportError:
    pass

try:
    import nmslib  # noqa pylint: disable=unused-import

    class NMSLibALSTest(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return NMSLibAlternatingLeastSquares(
                factors=32,
                regularization=0,
                index_params={"post": 2},
                random_state=23,
                use_gpu=False,
            )

        def test_pickle(self):
            # pickle isn't supported on nmslib indices
            pass

        def test_serialization(self):
            pass

        def test_serialization_without_fit(self):
            pass

    if HAS_CUDA:
        # nmslib doesn't support querying on the gpu, but we should be able to still use a GPU als
        # model with the nmslib index
        class NMSLibALSGPUTest(unittest.TestCase, RecommenderBaseTestMixin):
            def _get_model(self):
                return NMSLibAlternatingLeastSquares(
                    factors=32,
                    regularization=0,
                    index_params={"post": 2},
                    random_state=23,
                    use_gpu=True,
                )

            def test_pickle(self):
                # pickle isn't supported on nmslib indices
                pass

            def test_serialization(self):
                pass

            def test_serialization_without_fit(self):
                pass

except ImportError:
    pass

try:
    import faiss  # noqa pylint: disable=unused-import

    class FaissALSTest(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return FaissAlternatingLeastSquares(
                nlist=1, nprobe=1, factors=32, regularization=0, use_gpu=False, random_state=23
            )

        def test_pickle(self):
            # pickle isn't supported on faiss indices
            pass

        def test_serialization(self):
            pass

        def test_serialization_without_fit(self):
            pass

    if HAS_CUDA:

        class FaissALSGPUTest(unittest.TestCase, RecommenderBaseTestMixin):
            __regularization = 0

            def _get_model(self):
                return FaissAlternatingLeastSquares(
                    nlist=1,
                    nprobe=1,
                    factors=32,
                    regularization=self.__regularization,
                    use_gpu=True,
                    random_state=23,
                )

            def test_large_recommend(self):
                # the GPU version of FAISS can't return more than 1K result (and will assert/exit)
                # this tests out that we fall back in this case to the exact version and don't die
                plays = get_checker_board(2048)
                model = self._get_model()
                model.fit(plays, show_progress=False)

                recs = model.similar_items(0, N=1050)
                self.assertEqual(recs[0][0], 0)

                recs = model.recommend(0, plays.T.tocsr()[0], N=1050)
                self.assertEqual(recs[0][0], 0)

            def test_pickle(self):
                # pickle isn't supported on faiss indices
                pass

            def test_serialization(self):
                pass

            def test_serialization_without_fit(self):
                pass

except ImportError:
    pass

# mlxvs only installs on macOS with Apple silicon
try:
    import mlxvs  # noqa pylint: disable=unused-import

    class MLXVSALSTest(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return MLXVSAlternatingLeastSquares(factors=32, regularization=0, random_state=23)

        def test_large_recommend(self):
            # more results than the graph search returns: falls back to exact search
            plays = get_checker_board(2048)
            model = self._get_model()
            model.fit(plays, show_progress=False)

            recs = model.similar_items(0, N=1050)
            self.assertEqual(recs[0][0], 0)

            recs = model.recommend(0, plays.T.tocsr()[0], N=1050)
            self.assertEqual(recs[0][0], 0)

        def test_graph_index(self):
            # enough items to build the search graph: results match the exact model
            plays = get_checker_board(512)
            model = self._get_model()
            model.fit(plays, show_progress=False)
            self.assertIsNotNone(model.recommend_index.graph)

            ids, _ = model.similar_items(0, N=10)
            exact, _ = model.model.similar_items(0, N=10)
            self.assertGreaterEqual(len(set(ids) & set(exact)), 9)

        def test_pickle(self):
            # pickle isn't supported on mlxvs indices
            pass

        def test_serialization(self):
            pass

        def test_serialization_without_fit(self):
            pass

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
