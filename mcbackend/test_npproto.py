from datetime import datetime

import numpy
import pytest

from mcbackend import npproto


def ndarray_from_numpy(arr: numpy.ndarray) -> npproto.Ndarray:
    dt = str(arr.dtype)
    if "datetime64" in dt:
        # datetime64 doesn't support the buffer protocol.
        # See https://github.com/numpy/numpy/issues/4983
        # This is a hack that automatically encodes it as int64.
        arr = arr.astype("int64")
    return npproto.Ndarray(
        shape=list(arr.shape),
        dtype=dt,
        data=bytes(arr.data),
        strides=list(arr.strides),
    )


def ndarray_to_numpy(nda: npproto.Ndarray) -> numpy.ndarray:
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


@pytest.mark.parametrize(
    "arr",
    [
        numpy.arange(5),
        numpy.random.uniform(size=(2, 3)),
        numpy.array(5),
        numpy.array(["hello", "world"]),
        numpy.array([datetime(2020, 3, 4, 5, 6, 7, 8), datetime(2020, 3, 4, 5, 6, 7, 9)]),
        numpy.array([(1, 2), (3, 2, 1)], dtype=object),
    ],
)
def test_ndarray_protobuf(arr: numpy.ndarray):
    nda = ndarray_from_numpy(arr)
    enc = bytes(nda)
    dec = npproto.Ndarray().parse(enc)
    assert isinstance(dec.data, bytes)
    result = ndarray_to_numpy(dec)
    numpy.testing.assert_array_equal(result, arr)
    pass
