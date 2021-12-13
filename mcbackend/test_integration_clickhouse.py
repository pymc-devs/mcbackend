import clickhouse_driver
import hagelkorn
import pytest


@pytest.fixture
def cclient():
    # Set up a randomly named database for this test execution
    db = "testing_" + hagelkorn.random()
    main = clickhouse_driver.Client("localhost")
    main.execute(f"CREATE DATABASE {db};")
    # Give the test a client that targets the empty database
    client = clickhouse_driver.Client("localhost", database=db)
    yield client
    # Teardown
    client.disconnect()
    main.execute(f"DROP DATABASE {db};")
    main.disconnect()
    return


class TestClickHouseBackend:
    def test_test_database(self, cclient: clickhouse_driver.Client):
        assert cclient.execute("SHOW TABLES;") == []
        pass
