"""Tests for JSON output parser"""
import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator
from pydantic import BaseModel

from langprompt.output_parser.json import JSONOutputParser
from langprompt.base.response import Completion

class TestData(BaseModel):
    name: str
    age: int
    is_active: bool

def test_json_output_parser_dict():
    """Test JSONOutputParser with dict output"""
    parser = JSONOutputParser(dict)
    completion = Completion(
        content='{"name": "test", "value": 123}',
        role="assistant",
        id="test_id",
        created=int(datetime.now().timestamp()),
        model="test_model"
    )
    result = parser.parse(completion)
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 123

def test_json_output_parser_dataclass():
    """Test JSONOutputParser with dataclass output"""
    parser = JSONOutputParser(TestData)
    completion = Completion(
        content='{"name": "John", "age": 30, "is_active": true}',
        role="assistant",
        id="test_id",
        created=int(datetime.now().timestamp()),
        model="test_model"
    )
    result = parser.parse(completion)
    assert isinstance(result, TestData)
    assert result.name == "John"
    assert result.age == 30
    assert result.is_active is True

def test_json_output_parser_none_content():
    """Test JSONOutputParser with None content"""
    parser = JSONOutputParser(dict)
    completion = Completion(
        content=None,
        role="assistant",
        id="test_id",
        created=int(datetime.now().timestamp()),
        model="test_model"
    )
    with pytest.raises(ValueError, match="Completion content is None"):
        parser.parse(completion)

def test_json_output_parser_invalid_type():
    """Test JSONOutputParser with invalid type conversion"""
    parser = JSONOutputParser(TestData)
    completion = Completion(
        content='{"name": "John", "age": "not_a_number", "is_active": true}',
        role="assistant",
        id="test_id",
        created=int(datetime.now().timestamp()),
        model="test_model"
    )
    with pytest.raises(ValueError, match="Failed to convert JSON to TestData"):
        parser.parse(completion)

def test_json_output_parser_non_dict_json():
    """Test JSONOutputParser with non-dict JSON"""
    parser = JSONOutputParser(TestData)
    completion = Completion(
        content='[1, 2, 3]',
        role="assistant",
        id="test_id",
        created=int(datetime.now().timestamp()),
        model="test_model"
    )
    with pytest.raises(ValueError, match="Expected dict from JSON"):
        parser.parse(completion)

def test_json_output_parser_stream_not_supported():
    """Test JSONOutputParser stream_parse not supported"""
    parser = JSONOutputParser(dict)
    chunks: Iterator[Completion] = iter([])
    with pytest.raises(NotImplementedError, match="JSONOutputParser does not support streaming"):
        next(parser.stream_parse(chunks))