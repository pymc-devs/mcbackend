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
