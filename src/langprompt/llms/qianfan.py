"""A provider implementation for Baidu Qianfan API"""
from typing import Any, Dict, Sequence, Iterator, List, Literal, Optional
try:
    import qianfan
except ImportError:
    raise ImportError("Qianfan Python SDK is not installed. Please install it using `pip install qianfan`.")

# Local imports
from ..base.message import Message, TextPart, ImagePart
from ..base.response import Completion, CompletionUsage
from .base import BaseLLM
from ..store import BaseStore
from ..cache import BaseCache


class Qianfan(BaseLLM):
    """A provider implementation for Baidu Qianfan API"""

    def __init__(
        self,
        model: str = "ERNIE-Lite-8K-0922",
        temperature: float = 1.0,
        endpoint: Optional[str] = None,
        cache: Optional[BaseCache] = None,
        store: Optional[BaseStore] = None,
        query_per_second: float = 0,
        **kwargs
    ):
        """Initialize Qianfan provider

        Args:
            model: Model to use (e.g. ERNIE-Speed-8K, ERNIE-4.0-8K, etc.)
            temperature: Sampling temperature
            endpoint: Optional endpoint for custom models
            cache: Cache implementation
            **kwargs: Additional arguments to pass to the Qianfan client
        """
        super().__init__(cache=cache, store=store, query_per_second=query_per_second)
        chat_kwargs = kwargs.copy()
        chat_kwargs["model"] = model
        chat_kwargs["temperature"] = temperature
        if endpoint:
            chat_kwargs["endpoint"] = endpoint

        self.model = endpoint if endpoint else model

        self.chat_kwargs = chat_kwargs
        self.client = qianfan.ChatCompletion()

    def _chat(
        self,
        params: Dict[str, Any]
    ) -> Completion:
        """Send a chat completion request to Qianfan API

        Args:
            messages: List of Message objects
            **kwargs: Additional arguments to pass to the API

        Returns:
            Completion object
        """

        # Get response from Qianfan using do() method
        raw_response = self.client.do(**params)

        # Create Completion object
        return Completion(
            id=raw_response.body.get("id", ""), # type: ignore
            created=raw_response.body.get("created", 0), # type: ignore
            model=raw_response.body.get("model", self.model), # type: ignore
            usage=CompletionUsage(
                prompt_tokens=raw_response.body.get("usage", {}).get("prompt_tokens", 0), # type: ignore
                completion_tokens=raw_response.body.get("usage", {}).get("completion_tokens", 0), # type: ignore
                total_tokens=raw_response.body.get("usage", {}).get("total_tokens", 0) # type: ignore
            ),
            finish_reason=self._convert_finish_reason(raw_response.body), # type: ignore
            content=raw_response.body.get("result", ""), # type: ignore
            role="assistant",
            tool_calls=None,
            raw_response=raw_response.body, # type: ignore
        )

    def _stream(self, params: Dict[str, Any]) -> Iterator[Completion]:
        """Stream chat completion request from Qianfan API

        Args:
            messages: List of Message objects
            **kwargs: Additional arguments to pass to the API

        Returns:
            Iterator of Completion objects
        """

        # Get response from Qianfan using do() method
        raw_response = self.client.do(**params)

        for chunk in raw_response:
            yield Completion(
                id=chunk.body.get("id", ""),
                created=chunk.body.get("created", 0),
                model=chunk.body.get("model", self.model),
                usage=CompletionUsage(
                    prompt_tokens=chunk.body.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=chunk.body.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=chunk.body.get("usage", {}).get("total_tokens", 0)
                ) if chunk.body.get("usage") else None,
                finish_reason=self._convert_finish_reason(chunk.body),
                content=chunk.body.get("result", ""),
                role="assistant",
                tool_calls=None,
                raw_response=None,
            )

    def _prepare_params(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        """Prepare parameters for Qianfan API"""
        system = None
        qianfan_messages = []
        for message in messages:
            if message.role == "system":
                system = self._convert_content(message.content)
            else:
                qianfan_messages.append({"role": message.role, "content": self._convert_content(message.content)})

        params = {
            "messages": qianfan_messages,
            **self.chat_kwargs
        }
        if system:
            params["system"] = system # type: ignore
        params.update(kwargs)
        return params

    def _convert_content(self, content: str | Sequence[TextPart | ImagePart]) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, Sequence):
            # ignore image parts
            return "\n".join([part.text for part in content if isinstance(part, TextPart)])
        else:
            raise ValueError("Invalid content type")

    def _convert_finish_reason(self, body: Dict[str, Any]) -> Literal["stop", "length", "tool_calls", "content_filter"]:
        """Convert qianfan finish reason to openai finish reason
        Ref: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/xlmokikxe

        · normal：输出内容完全由大模型生成，未触发截断、替换
        · stop：输出结果命中入参stop中指定的字段后被截断
        · length：达到了最大的token数，根据EB返回结果is_truncated来截断
        · content_filter：输出内容被截断、兜底、替换为**等
        · function_call：调用了funtion call功能
        """
        finish_reason = body.get("finish_reason", "stop")
        finish_reason_map = {
            "normal": "stop",
            "stop": "stop",
            "length": "length",
            "content_filter": "content_filter",
            "function_call": "tool_calls",
        }
        ret = finish_reason_map.get(finish_reason, "stop")
        if body.get("need_clear_history", True):
            # 表示用户输入是否存在安全，是否关闭当前会话，清理历史会话信息。
            # true：是，表示用户输入存在安全风险，建议关闭当前会话，清理历史会话信息
            # false：否，表示用户输入无安全风险
            ret = "content_filter"
        if body.get("flag", 0):
            # 说明：返回flag表示触发安全
            ret = "content_filter"
        return ret # type: ignore
