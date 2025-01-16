import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional, cast

import duckdb

from .model import Record
from .store import BaseStore

__all__ = ["DuckDBStore"]

DEFAULT_DB_NAME = ".store.duckdb"


@dataclass
class DuckDBStore(BaseStore):
    """
    DuckDB store
    """

    path: str = field(
        default_factory=lambda: os.path.join(os.getcwd(), DEFAULT_DB_NAME)
    )
    _tables: List[str] = field(default_factory=list)
    _conn: Optional[duckdb.DuckDBPyConnection] = None

    def __post_init__(self):
        """Initialize database connection and table"""
        if self._conn is None:
            self._conn = duckdb.connect(self.path)
            self._tables = self._get_tables()

    @classmethod
    def connect(cls, path: Optional[str] = None) -> "DuckDBStore":
        """Create a new store instance with optional custom path"""
        if not path:
            path = os.path.join(os.getcwd(), DEFAULT_DB_NAME)
        return cls(path=path)

    def _init_table(self, record: Record) -> str:
        if not self._conn:
            raise RuntimeError("Database connection not initialized")

        table_name = f"{record.table_name}_{uuid.uuid4().hex[:8]}"
        assert isinstance(record.duckdb_schema, str), "duckdb_schema must be a string"
        assert isinstance(record.table_name, str), "table_name must be a string"
        duckdb_schema = record.duckdb_schema.replace(record.table_name, table_name)
        self._conn.execute(duckdb_schema)
        self._tables.append(table_name)
        return table_name

    def _get_tables(self) -> List[str]:
        """Get all tables in the database"""
        if not self._conn:
            return []
        return [table[0] for table in self._conn.execute("SHOW TABLES").fetchall()]

    def add(self, record: Record):
        """Add a new record to the database"""
        if not self._conn:
            raise RuntimeError("Database connection not initialized")

        assert isinstance(record.table_name, str), "table_name must be a string"
        table_name = next(
            (table for table in self._tables if record.table_name in table), None
        )
        if not table_name:
            table_name = self._init_table(record)

        record_dict = asdict(record)
        record_dict["synced_at"] = None

        assert hasattr(record, "json_fields"), "Record must have json_fields"
        json_fields = cast(List[str], record.json_fields)
        for json_field in json_fields:
            if record_dict[json_field]:
                record_dict[json_field] = json.dumps(
                    record_dict[json_field], ensure_ascii=False
                )

        assert hasattr(record, "table_columns"), "Record must have table_columns"
        table_columns = cast(List[str], record.table_columns)
        placeholders = ", ".join(["$" + str(i + 1) for i in range(len(table_columns))])

        # Sort record_dict based on table_columns order
        sorted_dict = {k: record_dict[k] for k in table_columns}
        record_dict = sorted_dict

        self._conn.execute(
            f"INSERT INTO {table_name} VALUES ({placeholders})",
            [record_dict[k] if k in record_dict else None for k in table_columns],
        )

    def get_unsynced(self, table_name: str) -> List[tuple]:
        """Retrieve unsynced records"""
        if not self._conn:
            raise RuntimeError("Database connection not initialized")

        return self._conn.execute(
            f"SELECT * FROM {table_name} WHERE synced_at IS NULL"
        ).fetchall()

    def mark_as_synced(self, record_ids: List[str], table_name: str) -> None:
        """Mark specified records as synced"""
        if not self._conn:
            raise RuntimeError("Database connection not initialized")

        self._conn.execute(
            f"UPDATE {table_name} SET synced_at = CURRENT_TIMESTAMP WHERE id = ANY($1)",
            [record_ids],
        )

    def close(self) -> None:
        """Close the database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
