from __future__ import print_function
import unittest
from scipy.sparse import csr_matrix
import numpy as np
import math

import implicit


class ImplicitALSTest(unittest.TestCase):
    def testImplicit(self):
        regularization = 1e-9
        tolerance = math.sqrt(regularization)
        tolerance = 0.001

        counts = csr_matrix([[1, 1, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [1, 0, 1, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0]], dtype=np.float64)

        def check_solution(rows, cols, counts):
            reconstructed = rows.dot(cols.T)
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    self.assertTrue(abs(counts[i, j] - reconstructed[i, j]) <
                                    tolerance)

        # try all 8 variants of native/python, cg/cholesky, and
        # 64 vs 32 bit factors
        for dtype in (np.float32, np.float64):
            for use_cg in (True, False):
                for use_native in (True, False):
                    rows, cols = implicit.alternating_least_squares(counts * 2, 7,
                                                                    regularization,
                                                                    use_native=use_native,
                                                                    use_cg=use_cg,
                                                                    dtype=dtype)
                    check_solution(rows, cols, counts.todense())

if __name__ == "__main__":
    unittest.main()
