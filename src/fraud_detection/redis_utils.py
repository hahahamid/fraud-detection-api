import logging
import time

import redis

logger = logging.getLogger(__name__)


def wait_for_redis(redis_client, max_retries: int = 10, initial_delay: float = 1.0) -> None:
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            redis_client.ping()
            return
        except redis.exceptions.ConnectionError:
            logger.error(
                "Redis unreachable (attempt %d/%d), retrying in %.1fs", attempt, max_retries, delay
            )
            time.sleep(delay)
            delay *= 2
    raise redis.exceptions.ConnectionError(f"Could not connect to Redis after {max_retries} attempts")
