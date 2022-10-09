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
        order="CF"[arr.flags.f_contiguous],
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
    # The code that reads the bytes(arr.data) always results in C-ordered data.
    # Passing `to the `np.ndarray(order=...)` call has no effect.
    # This is where below workaround comes into play:
    # It reshapes arrays that were originally in Fortran order.
    # F-ordered arrays that were encoded with mcbackend < 0.2.4 default
    # to `nda.order == ""` and there is nothing we can do for these.
    if nda.order == "F":
        arr = arr.T.reshape(arr.shape)
    return arr
