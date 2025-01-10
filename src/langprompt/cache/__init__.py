from abc import ABC, abstractmethod
import json
import sqlite3
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import threading
import time

class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

class MemoryCache(BaseCache):
    def __init__(self, ttl: int = 0):
        self._cache = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            if expiry and time.time() > expiry:
                del self._cache[key]
                return None

            return value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            expiry = None
            if self._ttl:
                expiry = time.time() + self._ttl
            self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

class SQLiteCache(BaseCache):
    def __init__(self, db_path: str = ".llm_cache.db", ttl: int = 0):
        self.db_path = db_path
        self._ttl = ttl
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def _init_db(self):
        with self._conn as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expiry TIMESTAMP
                )
            """)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._conn as conn:
            cursor = conn.execute(
                "SELECT value, expiry FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            value, expiry = row
            if expiry and datetime.now() > datetime.fromisoformat(expiry):
                self.delete(key)
                return None

            return json.loads(value)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._conn as conn:
            expiry = None
            if self._ttl:
                expiry = (datetime.now() + timedelta(seconds=self._ttl)).isoformat()

            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expiry) VALUES (?, ?, ?)",
                (key, json.dumps(value), expiry)
            )

    def delete(self, key: str) -> None:
        with self._conn as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear(self) -> None:
        with self._conn as conn:
            conn.execute("DELETE FROM cache")

    def __del__(self):
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn
