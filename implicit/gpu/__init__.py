from __future__ import absolute_import

try:
    import cupy  # noqa

    from ._cuda import *  # noqa

    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
