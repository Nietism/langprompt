import base64
import os
from pydantic import BaseModel
from langprompt import TextOutputParser, Prompt
from langprompt.llms.openai import OpenAI



class Input(BaseModel):
    image_base64: str

prompt = Prompt[Input]("""
<|system|>
You are a helpful assistant.
<|end|>

<|user|>
OCR the image: <|image|>{{ image_base64 }}<|/image|>
<|end|>
""")

parser = TextOutputParser()


if __name__ == "__main__":
    from langprompt.cache import SQLiteCache
    from langprompt.store import DuckDBStore
    provider = OpenAI(model="gpt-4o-mini", cache=SQLiteCache(), store=DuckDBStore())

    with open(os.path.join(os.path.dirname(__file__), "example.png"), "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    messages = prompt.parse(Input(image_base64=image_base64))
    print(f"Messages: {messages}")
    response = provider.chat(messages)
    print(f"Response: {response}")
    result = parser.parse(response)
    print(f"Result: {result}")
