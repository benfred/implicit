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

        # check cython version
        rows, cols = implicit.alternating_least_squares(counts * 2, 7,
                                                        regularization,
                                                        use_native=True)
        check_solution(rows, cols, counts.todense())

        # check cython version (using 32 bit factors)
        rows, cols = implicit.alternating_least_squares(counts * 2, 7,
                                                        regularization,
                                                        use_native=True,
                                                        dtype=np.float32)
        check_solution(rows, cols, counts.todense())

        # try out pure python version
        rows, cols = implicit.alternating_least_squares(counts, 7,
                                                        regularization,
                                                        use_native=False)
        check_solution(rows, cols, counts.todense())

if __name__ == "__main__":
    unittest.main()
