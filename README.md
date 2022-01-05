Where do _you_ want to store your MCMC draws?
In memory?
On disk?
Or in a database running in a datacenter?

No matter where you want to put them, or which <abbr title="probabilistic programming language">PPL</abbr> generates them: McBackend takes care of your MCMC samples.

## Quickstart
The `mcbackend` package consists of three parts:

### Part 1: A schema for MCMC run & chain metadata
No matter which programming language your favorite PPL is written in, the [ProtocolBuffers](https://developers.google.com/protocol-buffers/) from McBackend can be used to generate code in languages like C++, C#, Python and many more to represent commonly used metadata about MCMC runs, chains and model variables.

The definitions in [`protobufs/meta.proto`](./protobufs/meta.proto) are designed to maximize compatibility with [`ArviZ`](https://github.com/arviz-devs/arviz) objects, making it easy to transform MCMC draws stored according to the McBackend schema to `InferenceData` objects for plotting & analysis.

### Part 2: A storage backend interface
The  `draws` and `stats` created by MCMC sampling algorithms at runtime need to be stored _somewhere_.

This "somewhere" is called the storage _backend_ in PPLs/MCMC frameworks like [PyMC](https://github.com/pymc-devs/pymc) or [emcee](https://github.com/dfm/emcee).

Most storage backends must be initialized with metadata about the model variables so they can, for example, pre-allocated memory for the `draws` and `stats` they're about to receive.
After then receiving thousands of `draws` and `stats` they must then provide methods by which the `draws`/`stats` can be retrieved.

The `mcbackend.core` module has classes such as `Backend`, `Run`, and `Chain` to define these interfaces for any storage backend, no matter if it's an in-memory, filesystem or database storage.
Albeit this implementation is currently Python-only, the interface signature should be portable to e.g. C++.

Via `mcbackend.backends` the McBackend package then provides backend _implementations_.
Currently you may choose from:

```python
backend = mcbackend.NumPyBackend()
backend = mcbackend.ClickHouseBackend( client=clickhouse_driver.Client("localhost") )

# All that matters:
isinstance(backend, mcbackend.Backend)
# >>> True
```

### Part 3: PPL adapters
Anything that is a `Backend` can be wrapped by an [adapter ](https://en.wikipedia.org/wiki/Adapter_pattern) that makes it compatible with your favorite PPL.

In the example below, a `ClickHouseBackend` is initialized to store MCMC draws from a PyMC model in a [ClickHouse](http://clickhouse.com/) database.
See below for [how to run it in Docker](#development).

```python
import clickhouse_driver
import mcbackend
import pymc as pm

# 1. Create _any_ kind of backend
ch_client = clickhouse_driver.Client("localhost")
backend = mcbackend.ClickHouseBackend(ch_client)

with pm.Model():
    # 3. Create your model
    ...
    # 4. Wrap the PyMC adapter around an `mcbackend.Backend`
    #    This generates and prints a short `trace.run_id` by which
    #    this MCMC run is identified in the (database) backend.
    trace = mcbackend.pymc.TraceBackend(backend)

    # 5. Hit the inference button ™
    pm.sample(trace=trace)
```

Instead of using PyMC's built-in NumPy backend, the MCMC draws now end up in ClickHouse.

### Retrieving the `draws` & `stats`
Continuing the example from above we can now retrieve draws from the backend.

Note that since this example wrote the draws to ClickHouse, we could run the code below on another machine, and even while the above model is still sampling!

```python
backend = mcbackend.ClickHouseBackend(ch_client)

# Fetch the run from the database (downloads just metadata)
run = backend.get_run(trace.run_id)

# Get all draws from a chain
chain = run.get_chains()[0]
chain.get_draws("my favorite variable")
# >>> array([ ... ])

# Convert everything to `InferenceData`
idata = run.to_inferencedata()
print(idata)
# >>> Inference data with groups:
# >>> 	> posterior
# >>> 	> sample_stats
# >>>
# >>> Warmup iterations saved (warmup_*).
```

# Contributing what's next
McBackend just started and is looking for contributions.
For example:
* Schema discussion: Which metadata is needed? (related: [PyMC #5160](https://github.com/pymc-devs/pymc/issues/5160))
* Interface discussion: How should `Backend`/`Run`/`Chain` evolve?
* Python Backends for disk storage (HDF5, `*.proto`, ...)
* An `emcee` adapter (#11).
* C++ `Backend`/`Run`/`Chain` interfaces
* C++ ClickHouse backend (via [`clickhouse-cpp`](https://github.com/ClickHouse/clickhouse-cpp))
* A webinterface that goes beyond the Streamlit proof-of-concept (see [`mcbackend-server`](#experimental-arviz-server))

As the schema and API stabilizes a mid-term goal might be to replace PyMC `BaseTrace`/`MultiTrace` entirely to rely on `mcbackend`.

Getting rid of `MultiTrace` was a [long-term goal](https://github.com/pymc-devs/pymc/issues/4372#issuecomment-770100410) behind making `pm.sample(return_inferencedata=True)` the default.

## Development
First clone the repository and install `mcbackend` locally:

```bash
pip install -e .
```

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest -v
```

Some tests need a ClickHouse database server running locally.
To start one in Docker:

```bash
docker run --detach --rm --name mcbackend-db -p 9000:9000 --ulimit nofile=262144:262144 yandex/clickhouse-server
```

### Compiling the ProtocolBuffers
If you don't already have it, first install the protobuf compiler:
```bash
conda install protobuf
```

To compile the `*.proto` files for languages other than Python, check the [ProtocolBuffers documentation](https://developers.google.com/protocol-buffers/docs/tutorials).

The following script compiles them for Python using the [`betterproto`](https://github.com/danielgtaylor/python-betterproto) compiler plugin to get nice-looking dataclasses.
It also copies the generated files to the right place in `mcbackend`.

```bash
python protobufs/generate.py
```

# Experimental: `mcbackend-server`
This repository also includes an experimental Streamlit app for querying the ClickHouse backend and creating ArviZ plots already while an MCMC is still running.

⚠ This part will eventually move into its own repository. ⚠

First build the Docker image:

```
docker build -t mcbackend-server:0.1.0 .
```

Then start the container.
The following two commands should be executed in the root path of the repository.

⚠ You may need to adapt the hostname line. ⚠

On Windows:
```
docker run ^
  --rm --name mcbackend-server ^
  -p 8501:8501 ^
  -e DB_HOST=%COMPUTERNAME% ^
  -v %cd%:/mcbackend ^
  -v %cd%/mcbackend-server/app.py:/mcbackend-server/app.py ^
  mcbackend-server:0.1.0
```


On Linux:
```
docker run \
  --rm --name mcbackend-server \
  -p 8501:8501 \
  -e DB_HOST=$hostname \
  -v $pwd:/mcbackend \
  -v $pwd/mcbackend-server/app.py:/mcbackend-server/app.py \
  mcbackend-server:0.1.0
```
