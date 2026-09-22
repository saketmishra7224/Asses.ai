import subprocess
import time
import os
from redis import Redis


class _MemoryRedis:
    """Tiny in-memory stand-in so CLI scripts work without Docker/Redis."""

    def __init__(self):
        self._kv = {}
        self._counters = {}

    def ping(self):
        return True

    def get(self, key):
        return self._kv.get(key)

    def set(self, key, value, ex=None):
        self._kv[key] = value
        return True

    def delete(self, *keys):
        deleted = 0
        for k in keys:
            if k in self._kv:
                del self._kv[k]
                deleted += 1
            if k in self._counters:
                del self._counters[k]
        return deleted

    def exists(self, *keys):
        return sum(1 for k in keys if k in self._kv or k in self._counters)

    def incr(self, key):
        self._counters[key] = int(self._counters.get(key, 0)) + 1
        # mirror into kv for visibility
        self._kv[key] = str(self._counters[key])
        return self._counters[key]

    def keys(self, pattern="*"):
        import fnmatch

        return [k for k in self._kv.keys() if fnmatch.fnmatch(k, pattern)]

    def mget(self, keys, *args):
        all_keys = list(keys) if isinstance(keys, (list, tuple)) else [keys] + list(args)
        return [self._kv.get(k) for k in all_keys]

    def expire(self, key, time):
        return True

    def flushdb(self):
        self._kv.clear()
        self._counters.clear()
        return True

    def flushall(self):
        return self.flushdb()

    def close(self):
        pass


def start_redis_container():
    """Start Redis container if it's not already running (only for local development)."""
    # Skip if running inside Docker (REDIS_HOST env var is set)
    if os.environ.get("REDIS_HOST"):
        print("[redis] Running in Docker, skipping local Redis container start.")
        return

    try:
        # Check if Redis container is running
        result = subprocess.run(["docker", "ps", "-q", "--filter", "name=redis-stack"],
                                capture_output=True, text=True, timeout=10)
        if not result.stdout.strip():
            print("[redis] Starting Redis container...")
            subprocess.run([
                "docker", "run", "-d", "--name", "redis-stack",
                "-p", "6379:6379", "-p", "8001:8001", "redis/redis-stack:latest"
            ], check=True, timeout=120)
            print("[redis] Redis container started successfully.")
            time.sleep(5)  # Wait for Redis to initialize
        else:
            print("[redis] Redis container is already running.")
    except FileNotFoundError:
        print("[redis] Docker CLI not found; continuing without auto-start (in-memory fallback).")
    except Exception as e:
        print(f"[redis] Skipping auto-start ({type(e).__name__}: {e})")


# Start Redis container automatically (only for local dev)
start_redis_container()

# Get Redis connection details from environment variables (for Docker) or use defaults (for local)
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))


# Connect to Redis with retry logic
def connect_to_redis(max_retries=1, retry_delay=0.2):
    """Connect to Redis; fall back to in-memory store when unavailable."""
    import socket as _socket

    # Fast TCP probe first: avoids redis-py's long internal backoff when
    # nothing is listening (common on dev machines without Docker).
    try:
        probe = _socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=0.5)
        probe.close()
    except Exception as e:
        print(f"[redis] No server at {REDIS_HOST}:{REDIS_PORT} ({e}); using in-memory store.")
        return _MemoryRedis()

    for attempt in range(max_retries):
        try:
            client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
                           socket_connect_timeout=1, socket_timeout=1,
                           retry_on_timeout=False, health_check_interval=0)
            client.ping()
            print(f"[redis] Connected at {REDIS_HOST}:{REDIS_PORT}")
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[redis] Waiting for Redis... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"[redis] Unavailable ({e}); using in-memory store.")
                return _MemoryRedis()


redis_client = connect_to_redis()
