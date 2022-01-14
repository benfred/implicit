cdef extern from "implicit/gpu/utils.h" namespace "implicit::gpu" nogil:
    int get_device_count() except +
