import redis
import json
# from appconfig import config
from threading import Lock
import traceback

class RedisClient:
    _instance = None
    _lock = Lock()

    def __new__(cls, url, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                try:
                    cls._instance._r = redis.Redis.from_url(url)
                    cls._instance._r.ping()
                except:
                    return None
        return cls._instance
    
    def ping(self):
        try:
            self._r.ping()
            return True
        except:
            return False

    def set(self, key, value):
        """
        Set a value in Redis, handling serialization if the value is a dictionary.

        Args:
            key (str): The key under which to store the value.
            value (str or dict): The value to store. If a dictionary, it will be serialized to JSON.

        Returns:
            bool: True if the operation succeeded, False otherwise.
        """
        try:
            if isinstance(value, dict):
                # Serialize the dictionary to JSON
                value = json.dumps(value)
            # Store the value in Redis
            self._r.set(key, value)
            return True
        except Exception as e:
            print(f"Error setting value in Redis: {e}")
            print(traceback.format_exc())
            return False

    def get(self, key):
        """
        Get a value from Redis, handling deserialization if the value was a dictionary.

        Args:
            key (str): The key under which the value is stored in Redis.

        Returns:
            str or dict: The value retrieved from Redis. If the stored value was a dictionary, it's deserialized from JSON.
        """
        try:
            # Retrieve the value from Redis
            stored_value = self._r.get(key)
            if stored_value is not None:
                # Check if the stored value is a JSON string
                try:
                    stored_dict = json.loads(stored_value)
                    return stored_dict
                except json.JSONDecodeError:
                    # If it's not a JSON string, return as is
                    return stored_value.decode('utf-8')
            else:
                return None
        except Exception as e:
            print(f"Error getting value from Redis: {e}")
            print(traceback.format_exc())
            return None