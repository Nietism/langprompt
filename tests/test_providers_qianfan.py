"""Unit tests for QianfanProvider"""

import unittest

from langprompt.base.message import ImagePart, Message, TextPart
from langprompt.llms.qianfan import Qianfan


class TestQianfanProvider(unittest.TestCase):
    """Test cases for QianfanProvider"""

    def setUp(self):
        """Set up test cases"""
        self.provider = Qianfan(model="ERNIE-4.0-Turbo-8K", temperature=0.7)

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.provider.model, "ERNIE-4.0-Turbo-8K")

        # Test with endpoint
        provider = Qianfan(model="test-model", endpoint="custom-endpoint")
        self.assertEqual(provider.model, "custom-endpoint")

    def test_chat(self):
        """Test chat completion"""
        # Test messages
        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello"),
        ]

        # Get completion
        completion = self.provider.chat(messages)

        # Verify completion structure
        self.assertIsNotNone(completion.id)
        self.assertIsNotNone(completion.created)
        self.assertIsNotNone(completion.model)
        self.assertIsInstance(completion.content, str)
        self.assertEqual(completion.role, "assistant")
        self.assertIn(
            completion.finish_reason, ["stop", "length", "content_filter", "tool_calls"]
        )
        self.assertIsNotNone(completion.usage)
        if completion.usage:
            self.assertGreater(completion.usage.prompt_tokens, 0)
            self.assertGreater(completion.usage.completion_tokens, 0)
            self.assertGreater(completion.usage.total_tokens, 0)

    def test_stream(self):
        """Test streaming completion"""
        # Test messages
        messages = [Message(role="user", content="Hello")]

        # Get streaming completion
        completions = list(self.provider.stream(messages))

        # Verify completions
        self.assertGreater(len(completions), 0)
        for completion in completions:
            self.assertIsNotNone(completion.id)
            self.assertIsNotNone(completion.created)
            self.assertIsNotNone(completion.model)
            self.assertIsInstance(completion.content, str)
            self.assertEqual(completion.role, "assistant")
            self.assertIn(
                completion.finish_reason,
                ["stop", "length", "content_filter", "tool_calls"],
            )

        # 最后一个chunk应该包含usage信息
        self.assertIsNotNone(completions[-1].usage)
        if completions[-1].usage:
            self.assertGreater(completions[-1].usage.total_tokens, 0)

    def test_convert_content(self):
        """Test content conversion"""
        # Test string content
        content = "Hello world"
        self.assertEqual(self.provider._convert_content(content), "Hello world")

        # Test text parts
        content = [
            TextPart(text="Hello", type="text"),
            TextPart(text="world", type="text"),
        ]
        self.assertEqual(self.provider._convert_content(content), "Hello\nworld")

        # Test mixed content with image
        content = [
            TextPart(text="Hello", type="text"),
            ImagePart(image=b"fake_image_data", type="image", media_type="image/jpeg"),
            TextPart(text="world", type="text"),
        ]
        self.assertEqual(self.provider._convert_content(content), "Hello\nworld")

        # Test invalid content
        with self.assertRaises(ValueError):
            self.provider._convert_content(None)  # type: ignore

    def test_convert_finish_reason(self):
        """Test finish reason conversion"""
        # Test normal cases
        self.assertEqual(
            self.provider._convert_finish_reason(
                {"finish_reason": "normal", "need_clear_history": False, "flag": 0}
            ),
            "stop",
        )
        self.assertEqual(
            self.provider._convert_finish_reason(
                {"finish_reason": "stop", "need_clear_history": False, "flag": 0}
            ),
            "stop",
        )
        self.assertEqual(
            self.provider._convert_finish_reason(
                {"finish_reason": "length", "need_clear_history": False, "flag": 0}
            ),
            "length",
        )
        self.assertEqual(
            self.provider._convert_finish_reason(
                {
                    "finish_reason": "content_filter",
                    "need_clear_history": False,
                    "flag": 0,
                }
            ),
            "content_filter",
        )
        self.assertEqual(
            self.provider._convert_finish_reason(
                {
                    "finish_reason": "function_call",
                    "need_clear_history": False,
                    "flag": 0,
                }
            ),
            "tool_calls",
        )

        # Test need_clear_history
        self.assertEqual(
            self.provider._convert_finish_reason(
                {"finish_reason": "normal", "need_clear_history": True}
            ),
            "content_filter",
        )

        # Test flag
        self.assertEqual(
            self.provider._convert_finish_reason(
                {"finish_reason": "normal", "flag": 1}
            ),
            "content_filter",
        )

        # Test default
        self.assertEqual(
            self.provider._convert_finish_reason(
                {"finish_reason": "unknown", "need_clear_history": False, "flag": 0}
            ),
            "stop",
        )


if __name__ == "__main__":
    unittest.main()
