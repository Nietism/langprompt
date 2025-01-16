import base64
import os

from pydantic import BaseModel

from langprompt import Prompt, TextOutputParser
from langprompt.llms.openai import OpenAI


class Input(BaseModel):
    image: bytes

    @property
    def image_base64(self):
        return base64.b64encode(self.image).decode("utf-8")


class ImagePrompt(Prompt[Input, str]):
    def __init__(self):
        super().__init__(
            template="""
<|system|>
You are a helpful assistant.
<|end|>

<|user|>
OCR the image: <|image|>{{ input.image_base64 }}<|/image|>
<|end|>
""",
            output_parser=TextOutputParser(),
        )


if __name__ == "__main__":
    from langprompt.cache import SQLiteCache
    from langprompt.store import DuckDBStore

    provider = OpenAI(model="gpt-4o-mini", cache=SQLiteCache(), store=DuckDBStore())
    prompt = ImagePrompt()
    parser = TextOutputParser()

    with open(os.path.join(os.path.dirname(__file__), "example.png"), "rb") as f:
        image = f.read()

    messages = prompt.parse(Input(image=image))
    print(f"Messages: {messages}")
    response = provider.chat(messages)
    print(f"Response: {response}")
    result = prompt.parse_output(response)
    print(f"Result: {result}")
