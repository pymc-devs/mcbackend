import random

import hagelkorn
import numpy

from .core import Backend, Chain, ChainMeta, Run, RunMeta


def make_runmeta(*, flexibility: bool = False, **kwargs):
    defaults = dict(
        run_id=hagelkorn.random(),
        var_names=["tensor", "scalar"],
        var_dtypes=["int8", "float64"],
        var_shapes=[(3, 4, 5), ()],
        var_is_free=[True, False],
    )
    if flexibility:
        defaults["var_names"] = defaults["var_names"] + ["changeling"]
        defaults["var_dtypes"] = defaults["var_dtypes"] + ["uint16"]
        defaults["var_shapes"] = defaults["var_shapes"] + [(3, 0)]
        defaults["var_is_free"] = defaults["var_is_free"] + [True]
    defaults.update(kwargs)
    return RunMeta(**defaults)


def make_draw(names, shapes, dtypes):
    draw = {}
    for name, shape, dtype in zip(names, shapes, dtypes):
        dshape = tuple(
            # A pre-registered dim length of 0 means that it's random!
            s or random.randint(1, 10)
            for s in shape
        )
        if "float" in dtype:
            draw[name] = numpy.random.normal(size=dshape).astype(dtype)
        else:
            draw[name] = numpy.random.randint(low=0, high=100, size=dshape).astype(dtype)
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
        assert isinstance(chain.meta, ChainMeta)
        assert chain.meta.chain_number == 7
        assert isinstance(chain, Chain)
        assert isinstance(chain, self.cls_chain)
        pass

    def test__add_get_draw(self):
        rmeta = make_runmeta()
        run = self.backend.init_run(rmeta)
        chain = run.init_chain(7)
        expected = make_draw(rmeta.var_names, rmeta.var_shapes, rmeta.var_dtypes)
        chain.add_draw(expected)
        actual = chain.get_draw(0, rmeta.var_names)
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
        generated = [
            make_draw(rmeta.var_names, rmeta.var_shapes, rmeta.var_dtypes) for _ in range(10)
        ]
        for draw in generated:
            chain.add_draw(draw)

        # Fetch each variable once to check the returned type and values
        for vn, expdtype in zip(rmeta.var_names, rmeta.var_dtypes):
            expected = [draw[vn] for draw in generated]
            actual = chain.get_variable(vn)
            assert isinstance(actual, numpy.ndarray)
            if vn == "changeling":
                # Non-ridid variables are returned as object-arrays.
                assert actual.shape == (len(expected),)
                assert actual.dtype == object
                # Their values must be asserted elementwise to avoid shape problems.
                for act, exp in zip(actual, expected):
                    numpy.testing.assert_array_equal(act, exp)
            else:
                assert tuple(actual.shape) == tuple(numpy.shape(expected))
                assert actual.dtype == expdtype
                numpy.testing.assert_array_equal(actual, expected)
        pass
