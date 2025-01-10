"""A provider implementation for OpenAI API"""
from typing import Any, Dict, List, Iterator, Optional
import base64

try:
    import openai
except ImportError:
    raise ImportError("OpenAI Python SDK is not installed. Please install it using `pip install openai`.")

# Local imports
from ..base.message import Message, TextPart, ImagePart
from ..base.response import Completion, CompletionUsage
from ..store import BaseStore
from ..cache import BaseCache
from .base import BaseLLM

class OpenAI(BaseLLM):
    """A simplified OpenAI provider implementation"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        cache: Optional[BaseCache] = None,
        store: Optional[BaseStore] = None,
        query_per_second: float = 0,
        **kwargs
    ):
        """Initialize OpenAI provider

        Args:
            model: Model to use
            temperature: Sampling temperature
            cache: Cache implementation
            **kwargs: Additional arguments to pass to the OpenAI client
        """
        super().__init__(cache=cache, store=store, query_per_second=query_per_second)
        self.client = openai.OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature

    def _convert_message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert Message object to OpenAI API compatible dict

        Args:
            message: Message object

        Returns:
            Dict compatible with OpenAI API
        """
        if isinstance(message.content, str):
            return {"role": message.role, "content": message.content}

        # Handle multi-modal content
        content_parts = []
        for part in message.content:
            if isinstance(part, TextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{part.media_type};base64,{base64.b64encode(part.image).decode('utf-8')}",
                    }
                })
        return {"role": message.role, "content": content_parts}

    def _chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> Completion:
        """Send a chat completion request

        Args:
            messages: List of Message objects
            **kwargs: Additional arguments to pass to the API

        Returns:
            Completion object
        """
        params = self._prepare_params(messages, **kwargs)
        params["stream"] = False

        # Get response from OpenAI
        raw_response = self.client.chat.completions.create(**params)
        message = raw_response.choices[0].message

        # Create Completion object
        return Completion(
            id=raw_response.id,
            created=raw_response.created,
            model=raw_response.model,
            usage=CompletionUsage(
                prompt_tokens=raw_response.usage.prompt_tokens,
                completion_tokens=raw_response.usage.completion_tokens,
                total_tokens=raw_response.usage.total_tokens
            ),
            finish_reason=raw_response.choices[0].finish_reason,
            content=None if getattr(message, "tool_calls", None) else message.content,
            role=message.role,
            tool_calls=message.tool_calls,
            raw_response=raw_response.model_dump()
        )

    def _stream(self, messages: List[Message], **kwargs) -> Iterator[Completion]:
        """Stream chat completion request

        Args:
            messages: List of Message objects
            **kwargs: Additional arguments to pass to the API

        Returns:
            Iterator of Completion objects
        """
        params = self._prepare_params(messages, **kwargs)
        params["stream"] = True

        # Get response from OpenAI
        raw_response = self.client.chat.completions.create(**params)

        for chunk in raw_response:
            delta = chunk.choices[0].delta
            yield Completion(
                id=chunk.id,
                created=chunk.created,
                model=chunk.model,
                usage=CompletionUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                ) if chunk.usage else None,
                finish_reason=chunk.choices[0].finish_reason,
                content=delta.content if hasattr(delta, "content") else "",
                role=delta.role if hasattr(delta, "role") else None,
                tool_calls=delta.tool_calls if hasattr(delta, "tool_calls") else None,
                raw_response=None
            )

    def _prepare_params(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        """Prepare parameters for OpenAI API"""
        openai_messages = [
            self._convert_message_to_dict(msg) for msg in messages
        ]
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": openai_messages,
        }
        params.update(kwargs)
        return params
