"""
Helper functions such as converters between ``ndarray`` and ``Ndarray``.
"""
import numpy

from . import Ndarray


def ndarray_from_numpy(arr: numpy.ndarray) -> Ndarray:
    dt = str(arr.dtype)
    if "datetime64" in dt:
        # datetime64 doesn't support the buffer protocol.
        # See https://github.com/numpy/numpy/issues/4983
        # This is a hack that automatically encodes it as int64.
        arr = arr.astype("int64")
    return Ndarray(
        shape=list(arr.shape),
        dtype=dt,
        data=bytes(arr.data),
        strides=list(arr.strides),
    )


def ndarray_to_numpy(nda: Ndarray) -> numpy.ndarray:
    arr: numpy.ndarray
    if "datetime64" in nda.dtype:
        # Backwards conversion: The data was stored as int64.
        arr = numpy.ndarray(
            buffer=nda.data,
            shape=nda.shape,
            dtype="int64",
            strides=nda.strides,
        ).astype(nda.dtype)
    else:
        arr = numpy.ndarray(
            buffer=nda.data,
            shape=nda.shape,
            dtype=numpy.dtype(nda.dtype),
            strides=nda.strides,
        )
    return arr
