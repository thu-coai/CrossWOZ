from functools import wraps
import redis
from redis_lock import Lock
from dotenv import load_dotenv
from os import getenv

load_dotenv()
rd = redis.from_url(getenv('REDIS_URL', 'redis://localhost:6379/0'))


def with_lock(name):
    def decorator(f):
        wraps(f)

        def decorated(*args, **kwargs):
            with Lock(rd, name):
                res = f(*args, **kwargs)
            return res

        return decorated

    return decorator
