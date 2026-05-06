from __future__ import absolute_import

HAS_CUDA = False
try:
    from ._cuda import *  # noqa

    get_device_count()  # noqa pylint: disable=undefined-variable
    HAS_CUDA = True

except RuntimeError as e:
    import warnings

    warnings.warn(
        f"CUDA extension is built, but disabling GPU support because of '{e}'",
    )
except ImportError:
    pass
