import random
from typing import Sequence

import hagelkorn
import numpy

from mcbackend.meta import ChainMeta, RunMeta, Variable

from .core import Backend, Chain, Run


def make_runmeta(*, flexibility: bool = False, **kwargs) -> RunMeta:
    defaults = dict(
        rid=hagelkorn.random(),
        variables=[
            Variable("tensor", "int8", [3, 4, 5], True),
            Variable("scalar", "float64", [], False),
        ],
    )
    if flexibility:
        defaults["variables"].append(Variable("changeling", "uint16", [3, 0], True))
    defaults.update(kwargs)
    return RunMeta(**defaults)


def make_draw(variables: Sequence[Variable]):
    draw = {}
    for var in variables:
        dshape = tuple(
            # A pre-registered dim length of 0 means that it's random!
            s or random.randint(0, 10)
            for s in var.shape
        )
        if "float" in var.dtype:
            draw[var.name] = numpy.random.normal(size=dshape).astype(var.dtype)
        else:
            draw[var.name] = numpy.random.randint(low=0, high=100, size=dshape).astype(var.dtype)
    return draw


class BaseBackendTest:
    """Can be used to test different backends in the same way."""

    cls_backend = None
    cls_run = None
    cls_chain = None

    def setup_method(self, method):
        """Override this when the backend has no parameterless constructor."""
        self.backend = self.cls_backend()

    def teardown_method(self, method):
        pass


class CheckBehavior(BaseBackendTest):
    """Validates that a backend shows the expected behavior via the common API."""

    def test__initialization(self):
        assert isinstance(self.backend, Backend)
        assert isinstance(self.backend, self.cls_backend)
        rmeta = make_runmeta()
        run = self.backend.init_run(rmeta)
        assert isinstance(run, Run)
        assert isinstance(run, self.cls_run)
        chain = run.init_chain(7)
        assert isinstance(chain.cmeta, ChainMeta)
        assert chain.cmeta.chain_number == 7
        assert isinstance(chain, Chain)
        assert isinstance(chain, self.cls_chain)
        pass

    def test__add_get_draw(self):
        rmeta = make_runmeta()
        run = self.backend.init_run(rmeta)
        chain = run.init_chain(7)
        expected = make_draw(rmeta.variables)
        chain.add_draw(expected)
        actual = chain.get_draw(0, [v.name for v in rmeta.variables])
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for vn, act in actual.items():
            numpy.testing.assert_array_equal(act, expected[vn])
        pass

    def test__get_variable(self):
        rmeta = make_runmeta(flexibility=True)
        run = self.backend.init_run(rmeta)
        chain = run.init_chain(7)

        # Generate draws and add them to the chain
        generated = [make_draw(rmeta.variables) for _ in range(10)]
        for draw in generated:
            chain.add_draw(draw)

        # Fetch each variable once to check the returned type and values
        for var in rmeta.variables:
            expected = [draw[var.name] for draw in generated]
            actual = chain.get_variable(var.name)
            assert isinstance(actual, numpy.ndarray)
            if var.name == "changeling":
                # Non-ridid variables are returned as object-arrays.
                assert actual.shape == (len(expected),)
                assert actual.dtype == object
                # Their values must be asserted elementwise to avoid shape problems.
                for act, exp in zip(actual, expected):
                    numpy.testing.assert_array_equal(act, exp)
            else:
                assert tuple(actual.shape) == tuple(numpy.shape(expected))
                assert actual.dtype == var.dtype
                numpy.testing.assert_array_equal(actual, expected)
        pass


class CheckPerformance(BaseBackendTest):
    """Checks that the backend is reasonably fast via various high-load tests."""

    def test__many_draws(self):
        pass

    def test__many_variables(self):
        pass

    def test__big_variables(self):
        pass
