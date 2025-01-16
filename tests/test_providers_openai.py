"""Tests for OpenAI provider"""

import base64

import pytest

from langprompt.base.message import ImagePart, Message, TextPart
from langprompt.llms.openai import OpenAI


@pytest.fixture
def provider():
    return OpenAI(model="gpt-4", temperature=0.7)


def test_init():
    """Test provider initialization"""
    provider = OpenAI(model="gpt-4", temperature=0.7)
    assert provider.model == "gpt-4"
    assert provider.temperature == 0.7


def test_convert_text_message():
    """Test converting text message to OpenAI format"""
    provider = OpenAI()
    message = Message(role="user", content="Hello")
    result = provider._convert_message_to_dict(message)
    assert result == {"role": "user", "content": "Hello"}


def test_convert_multimodal_message():
    """Test converting multimodal message to OpenAI format"""
    provider = OpenAI()
    image_data = b"fake_image_data"
    message = Message(
        role="user",
        content=[
            TextPart(text="Hello", type="text"),
            ImagePart(image=image_data, media_type="image/jpeg", type="image"),
        ],
    )
    result = provider._convert_message_to_dict(message)

    expected_image_url = (
        f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
    )
    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": expected_image_url}},
        ],
    }


def test_chat():
    """Test chat completion"""
    provider = OpenAI()
    messages = [Message(role="user", content="Hi")]
    completion = provider.chat(messages)

    # 验证基本响应结构
    assert isinstance(completion.content, str)
    assert completion.role == "assistant"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.completion_tokens > 0
    assert completion.usage.total_tokens > 0
    assert completion.finish_reason in ["stop", "length"]
    assert completion.tool_calls is None


def test_stream():
    """Test streaming completion"""
    provider = OpenAI()
    messages = [Message(role="user", content="Hi")]
    response_chunks = list(provider.stream(messages))

    # 验证流式响应
    assert len(response_chunks) > 0
    full_response = "".join(chunk.content for chunk in response_chunks if chunk.content)
    assert isinstance(full_response, str)
    assert len(full_response) > 0
    assert response_chunks[0].role == "assistant"


def test_prepare_params():
    """Test parameter preparation"""
    provider = OpenAI(model="gpt-4", temperature=0.7)
    messages = [Message(role="user", content="Hello")]
    params = provider._prepare_params(messages, max_tokens=100)

    assert params["model"] == "gpt-4"
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 100
    assert len(params["messages"]) == 1
    assert params["messages"][0] == {"role": "user", "content": "Hello"}
