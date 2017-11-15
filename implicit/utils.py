import logging
import os

import numpy as np


def nonzeros(m, row):
    """ returns the non zeroes of a row in csr_matrix """
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]


def check_blas_config():
    """ checks to see if using OpenBlas/Intel MKL. If so, warn if the number of threads isn't set
    to 1 (causes severe perf issues when training - can be 10x slower) """
    if np.__config__.get_info('openblas_info') and os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        logging.warning("OpenBLAS detected. Its highly recommend to set the environment variable "
                        "'export OPENBLAS_NUM_THREADS=1' to disable its internal multithreading")
    if np.__config__.get_info('blas_mkl_info') and os.environ.get('MKL_NUM_THREADS') != '1':
        logging.warning("Intel MKL BLAS detected. Its highly recommend to set the environment "
                        "variable 'export MKL_NUM_THREADS=1' to disable its internal "
                        "multithreading")
