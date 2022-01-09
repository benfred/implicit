from libcpp cimport bool
from libcpp.utility cimport pair

from .matrix cimport Matrix, Vector


cdef extern from "implicit/gpu/bpr.h" namespace "implicit::gpu" nogil:
    cdef pair[int, int] bpr_update(const Vector[int] & userids,
                                   const Vector[int] & itemids,
                                   const Vector[int] & indptr,
                                   Matrix * X,
                                   Matrix * Y,
                                   float learning_rate, float regularization, long seed,
                                   bool verify_negative_samples) except +
