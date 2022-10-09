from datetime import datetime

import numpy
import pytest

from mcbackend import npproto
from mcbackend.npproto import utils


class TestUtils:
    @pytest.mark.parametrize(
        "arr",
        [
            numpy.arange(5),
            numpy.random.uniform(size=(2, 3)),
            numpy.array(5),
            numpy.array(["hello", "world"]),
            numpy.array([datetime(2020, 3, 4, 5, 6, 7, 8), datetime(2020, 3, 4, 5, 6, 7, 9)]),
            numpy.array(
                [datetime(2020, 3, 4, 5, 6, 7, 8), datetime(2020, 3, 4, 5, 6, 7, 9)],
                dtype="datetime64",
            ),
            numpy.array([(1, 2), (3, 2, 1)], dtype=object),
        ],
    )
    def test_conversion(self, arr: numpy.ndarray):
        nda = utils.ndarray_from_numpy(arr)
        enc = bytes(nda)
        dec = npproto.Ndarray().parse(enc)
        assert isinstance(dec.data, bytes)
        result = utils.ndarray_to_numpy(dec)
        numpy.testing.assert_array_equal(result, arr)
        pass

    @pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 5), (5, 2, 1, 7)])
    @pytest.mark.parametrize("order", "CF")
    def test_byteorders(self, shape, order):
        arr = numpy.arange(numpy.prod(shape)).reshape(shape, order=order)

        nda = utils.ndarray_from_numpy(arr)
        assert nda.order == "CF"[arr.flags.f_contiguous]

        dec = utils.ndarray_to_numpy(nda)
        numpy.testing.assert_array_equal(arr, dec)
        pass
