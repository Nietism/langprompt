import datetime
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langprompt.base.message import Message
from langprompt.base.response import Completion


@dataclass
class Record(ABC):
    """
    Base class for storing model response information
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str = field(default="")
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    raw_response: Optional[Dict] = None

    @property
    @abstractmethod
    def json_fields(self):
        """Return the DuckDB JSON fields for the record"""
        pass

    @property
    @abstractmethod
    def image_fields(self):
        """Return the DuckDB image fields for the record"""
        pass

    @property
    @abstractmethod
    def table_columns(self):
        """Return the DuckDB table columns for the record"""
        pass

    @property
    @abstractmethod
    def duckdb_schema(self):
        """Return the DuckDB schema for the record"""
        pass

    @property
    @abstractmethod
    def table_name(self):
        """Return the DuckDB table name for the record"""
        pass


@dataclass
class ResponseRecord(Record):
    """
    Data class for storing API response information
    """

    response_id: str = field(default="")
    model: str = field(default="")
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    messages: List[Dict[str, str]] = field(default_factory=list)
    assistant_message: Optional[str] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[Any] = None
    tags: Optional[List[str]] = field(default_factory=list)
    properties: Optional[Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    synced_at: Optional[str] = None

    @classmethod
    def create(cls, messages: List[Message], response: Optional[Completion] = None, error: Optional[Exception] = None, **kwargs):
        """Create a response record from an API response or error"""
        if not response:
            return cls(finish_reason="error", error=str(error), **kwargs)

        usage = response.usage.model_dump() if response.usage else {}

        return cls( 
            response_id=response.id,
            completion_tokens=usage.get("completion_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            total_tokens=usage.get("total_tokens"),
            assistant_message=response.content,
            finish_reason=response.finish_reason,
            tool_calls=response.tool_calls,
            messages=[msg.model_dump() for msg in messages],
            raw_response=response.raw_response,
            **kwargs,
        )

    @property
    def table_columns(self):
        return [
            "id",
            "response_id",
            "model",
            "timestamp",
            "messages",
            "assistant_message",
            "completion_tokens",
            "prompt_tokens",
            "total_tokens",
            "finish_reason",
            "tool_calls",
            "tags",
            "properties",
            "error",
            "raw_response",
            "synced_at",
        ]

    @property
    def duckdb_schema(self):
        return f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            response_id VARCHAR,
            model VARCHAR,
            timestamp TIMESTAMP,
            messages STRUCT(role VARCHAR, content VARCHAR)[],
            assistant_message TEXT,
            completion_tokens INTEGER,
            prompt_tokens INTEGER,
            total_tokens INTEGER,
            finish_reason VARCHAR,
            tool_calls JSON,
            tags VARCHAR[],
            properties JSON,
            error VARCHAR,
            raw_response JSON,
            synced_at TIMESTAMP
        );
        CREATE INDEX response_id_idx ON {self.table_name} (response_id);
        CREATE INDEX model_idx ON {self.table_name} (model);
        CREATE INDEX finish_reason_idx ON {self.table_name} (finish_reason);
        """

    @property
    def table_name(self):
        return "records"

    @property
    def json_fields(self):
        return ["tool_calls", "tags", "properties", "raw_response"]

    @property
    def image_fields(self):
        return []

    @property
    def text_fields(self):
        return []
