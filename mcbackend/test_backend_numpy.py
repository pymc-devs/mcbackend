import random

import hagelkorn
import numpy
import pytest

from mcbackend.backends.numpy import NumPyBackend, NumPyChain, NumPyRun
from mcbackend.core import RunMeta
from mcbackend.meta import Variable
from mcbackend.test_utils import CheckBehavior, CheckPerformance


class TestNumPyBackend(CheckBehavior, CheckPerformance):
    cls_backend = NumPyBackend
    cls_run = NumPyRun
    cls_chain = NumPyChain

    def test_targets(self):
        imb = NumPyBackend(preallocate=123)
        rm = RunMeta(
            rid=hagelkorn.random(),
            variables=[
                Variable("tensor", "int8", (3, 4, 5)),
                Variable("scalar", "float64", ()),
                Variable("changeling", "uint16", (3, -1)),
            ],
        )
        run = imb.init_run(rm)
        chain = run.init_chain(0)
        # Shape flexibility detection
        assert chain._var_is_rigid["tensor"]
        assert chain._var_is_rigid["scalar"]
        assert not chain._var_is_rigid["changeling"]
        # Types of targets
        assert isinstance(chain._samples["tensor"], numpy.ndarray)
        assert isinstance(chain._samples["scalar"], numpy.ndarray)
        assert isinstance(chain._samples["changeling"], numpy.ndarray)
        # Shapes and dtypes
        assert chain._samples["tensor"].shape == (123, 3, 4, 5)
        assert chain._samples["scalar"].shape == (123,)
        assert chain._samples["changeling"].shape == (123,)
        assert chain._samples["tensor"].dtype == "int8"
        assert chain._samples["scalar"].dtype == "float64"
        assert chain._samples["changeling"].dtype == object
        pass

    @pytest.mark.parametrize("preallocate", [0, 75])
    def test_growing(self, preallocate):
        imb = NumPyBackend(preallocate=preallocate)
        rm = RunMeta(
            rid=hagelkorn.random(),
            variables=[
                Variable(
                    "A",
                    "float32",
                    (2,),
                ),
                Variable(
                    "B",
                    "float32",
                    (-1,),
                ),
            ],
        )
        run = imb.init_run(rm)
        chain = run.init_chain(0)
        assert chain._samples["A"].shape == (preallocate, 2)
        assert chain._samples["B"].shape == (preallocate,)
        for _ in range(130):
            draw = {
                "A": numpy.random.uniform(size=(2,)),
                "B": numpy.random.uniform(size=(random.randint(0, 10),)),
            }
            chain.append(draw)
        # NB: Growth algorithm adds max(10, ceil(0.1*length)) 
        if preallocate == 75:
            # 75 → 85 → 95 → 105 → 116 → 128 → 141
            assert chain._samples["A"].shape == (141, 2)
            assert chain._samples["B"].shape == (141,)
        elif preallocate == 0:
            # 10 → 20 → ... → 90 → 100 → 110 → 121 → 134
            assert chain._samples["A"].shape == (134, 2)
            assert chain._samples["B"].shape == (134,)
        else:
            assert False, f"Missing test for {preallocate=}"
        assert chain.get_draws("A").shape == (130, 2)
        assert chain.get_draws("B").shape == (130,)
        pass


if __name__ == "__main__":
    tc = TestNumPyBackend()
    df = tc.run_all_benchmarks()
    print(df)
