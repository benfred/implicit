from __future__ import print_function

import unittest

import numpy as np
from scipy.sparse import csr_matrix, random

from implicit.evaluation import leave_k_out_split, train_test_split


class EvaluationTest(unittest.TestCase):

    @staticmethod
    def _get_sample_matrix():
        return csr_matrix((np.random.random((10, 10)) > 0.5).astype(np.float64))

    @staticmethod
    def _get_matrix():
        mat = random(100, 100, density=0.5, format="csr", dtype=np.float32)
        return mat.tocoo()

    def test_train_test_split(self):
        seed = np.random.randint(1000)
        mat = self._get_sample_matrix()
        train, test = train_test_split(mat, 0.8, seed)
        train2, test2 = train_test_split(mat, 0.8, seed)
        self.assertTrue(np.all(train.todense() == train2.todense()))

    def test_leave_k_out_returns_correct_shape(self):
        """
        Test that the output matrices are of the same shape as the input matrix.
        """

        mat = self._get_matrix()
        train, test = leave_k_out_split(mat, K=1)
        self.assertTrue(train.shape == mat.shape)
        self.assertTrue(test.shape == mat.shape)

    def test_leave_k_out_outputs_produce_input(self):
        """
        Test that the sum of the output matrices is equal to the input matrix (i.e.
        that summing the output matrices produces the input matrix).
        """

        mat = self._get_matrix()
        train, test = leave_k_out_split(mat, K=1)
        self.assertTrue(((train + test) - mat).nnz == 0)

    def test_leave_k_split_is_reservable(self):
        """
        Test that the sum of the train and test set equals the input.
        """

        mat = self._get_matrix()
        train, test = leave_k_out_split(mat, K=1)

        # check all matrices are positive, non-zero
        self.assertTrue(mat.sum() > 0)
        self.assertTrue(test.sum() > 0)
        self.assertTrue(train.sum() > 0)

        # check sum of train + test = input
        self.assertTrue(((train + test) - mat).nnz == 0)

    def test_leave_k_out_gets_correct_train_only_shape(self):
        """Test that the correct number of users appear *only* in the train set."""

        mat = self._get_matrix()
        train, test = leave_k_out_split(mat, K=1, train_only_size=0.8)
        train_only = ~np.isin(np.unique(train.tocoo().row), test.tocoo().row)
        self.assertTrue(train_only.sum() == int(train.shape[0] * 0.8))

    def test_leave_k_out_raises_error_for_k_less_than_zero(self):
        """
        Test that an error is raised when K < 0.
        """

        self.assertRaises(ValueError, leave_k_out_split, None, K=0)

    def test_leave_k_out_raises_error_for_invalid_train_only_size_lower_bound(self):
        """
        Test that an error is raised when train_only_size < 0.
        """

        self.assertRaises(
            ValueError, leave_k_out_split, None, K=1, train_only_size=-1.0
        )

    def test_leave_k_out_raises_error_for_invalid_train_only_size_upper_bound(self):
        """
        Test that an error is raised when train_only_size >= 1.
        """

        self.assertRaises(
            ValueError, leave_k_out_split, None, K=1, train_only_size=1.0
        )


if __name__ == "__main__":
    unittest.main()
