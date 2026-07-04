import pytest
import redis

from fraud_detection.redis_utils import wait_for_redis


class _AlwaysFailsPing:
    def ping(self):
        raise redis.exceptions.ConnectionError("unreachable")


class _SucceedsPing:
    def __init__(self):
        self.calls = 0

    def ping(self):
        self.calls += 1
        return True


def test_wait_for_redis_returns_immediately_when_reachable():
    client = _SucceedsPing()
    wait_for_redis(client, max_retries=3, initial_delay=0.01)
    assert client.calls == 1


def test_wait_for_redis_raises_after_max_retries():
    client = _AlwaysFailsPing()
    with pytest.raises(redis.exceptions.ConnectionError):
        wait_for_redis(client, max_retries=3, initial_delay=0.01)
