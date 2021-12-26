# McBackend: Framework-agnostic storage of MCMC draws
The `mcbackend` package consists of three parts:
1. A schema for MCMC sample storage backends: `mcbackend.core`
2. Backend implementations: `mcbackend.backends`
3. Adapters for specific PPLs: `mcbackend.adapters`

One of the backend implementations uses [ClickHouse](https://github.com/ClickHouse/ClickHouse) to store draws.
To run the integration tests in the test suite, first start a ClickHouse database server in Docker using the following command:

```bash
docker run --detach --rm --name arviz-db -p 9000:9000 --ulimit nofile=262144:262144 yandex/clickhouse-server
```

After that just run `pytest -vx`.

## Compiling the ProtocolBuffers
If you don't already have it, first install the protobuf compiler:
```bash
conda install protobuf
```

Then compile the `*.proto` files:
```bash
cd protobufs
python generate.py
```

# Experimental: `ArviZ-server`
This repository also includes an experimental Streamlit app for querying the ClickHouse backend and ArviZ plots while the MCMC is still running.

⚠ This part will eventually be moved into its own repository. ⚠

First build the Docker image:

```
docker build -t arviz-server:0.1.0 .
```

Then start the container.
The following two commands should be executed in the root path of the repository.
You may need to adapt the hostname line.

On Windows:
```
docker run ^
  --rm --name arviz-server ^
  -p 8501:8501 ^
  -e ARVIZ_DB_HOST=%COMPUTERNAME%.fritz.box ^
  -v %cd%:/mcbackend ^
  -v %cd%/arviz-server/app.py:/arviz-server/app.py ^
  arviz-server:0.1.0
```


On Linux:
```
docker run \
  --rm --name arviz-server \
  -p 8501:8501 \
  -e ARVIZ_DB_HOST=$hostname.fritz.box \
  -v $pwd:/mcbackend \
  -v $pwd/arviz-server/app.py:/arviz-server/app.py \
  arviz-server:0.1.0
```
