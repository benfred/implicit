from __future__ import absolute_import

import warnings

HAS_CUDA = False
HAS_RMM = False

try:
    # RMM is required to enable GPU support - use with 'pip install rmm-cu13'
    import rmm  # noqa

    HAS_RMM = True

    from ._cuda import *  # noqa

    get_device_count()  # noqa pylint: disable=undefined-variable
    HAS_CUDA = True

except RuntimeError as e:
    import warnings

    warnings.warn(
        f"CUDA extension is built, but disabling GPU support because of '{e}'",
    )
except ImportError as e:
    if HAS_RMM:
        warnings.warn(
            f"Disabling GPU support because of '{e}'",
        )
