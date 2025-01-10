import os
import tempfile
from unittest import TestCase

import duckdb
from langprompt.trace.duckdb import DuckDBStore
from langprompt.trace.model import ResponseRecord

class TestDuckDBStore(TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.duckdb")
        self.store = DuckDBStore.connect(self.db_path)
        
    def tearDown(self):
        """Clean up test environment after each test"""
        self.store.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
        
    def test_connection(self):
        """Test database connection and initialization"""
        self.assertIsNotNone(self.store._conn)
        self.assertIsInstance(self.store._conn, duckdb.DuckDBPyConnection)
        
    def test_add_record(self):
        """Test adding a record to the database"""
        record = ResponseRecord(
            id="test1",
            model="test-model",
            assistant_message="test content",
            messages=[{"role": "user", "content": "test message"}]
        )
        self.store.add(record)
        
        # Verify record was added
        result = self.store._conn.execute(
            f"SELECT id, model, assistant_message FROM {self.store._tables[0]} WHERE id = 'test1'"
        ).fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "test1")
        self.assertEqual(result[1], "test-model")
        self.assertEqual(result[2], "test content")
        
    def test_get_unsynced(self):
        """Test retrieving unsynced records"""
        # Add test records
        records = [
            ResponseRecord(id="test1", model="model1", assistant_message="content1"),
            ResponseRecord(id="test2", model="model2", assistant_message="content2")
        ]
        for record in records:
            self.store.add(record)
            
        table_name = self.store._tables[0]
        unsynced = self.store.get_unsynced(table_name)
        
        self.assertEqual(len(unsynced), 2)
        self.assertEqual(unsynced[0][0], "test1")
        self.assertEqual(unsynced[1][0], "test2")
        
    def test_mark_as_synced(self):
        """Test marking records as synced"""
        # Add test record
        record = ResponseRecord(id="test1", model="test-model", assistant_message="test content")
        self.store.add(record)
        
        table_name = self.store._tables[0]
        self.store.mark_as_synced(["test1"], table_name)
        
        # Verify record is marked as synced
        unsynced = self.store.get_unsynced(table_name)
        self.assertEqual(len(unsynced), 0)
        
        # Verify synced_at is set
        result = self.store._conn.execute(
            f"SELECT synced_at FROM {table_name} WHERE id = 'test1'"
        ).fetchone()
        self.assertIsNotNone(result[0])
        
    def test_close_connection(self):
        """Test closing database connection"""
        self.store.close()
        self.assertIsNone(self.store._conn)
        
        # Test context manager
        with DuckDBStore.connect(self.db_path) as store:
            self.assertIsNotNone(store._conn)
        self.assertIsNone(store._conn) 