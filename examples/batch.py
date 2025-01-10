from pydantic import BaseModel
from langprompt import TextOutputParser, Prompt
from langprompt.llms.openai import OpenAI
from langprompt.cache import SQLiteCache
from langprompt.store import DuckDBStore


class Input(BaseModel):
    text: str
    language: str = "Chinese"

prompt = Prompt[Input]("""
<|system|>
You are a professional translator. Please accurately translate the text while maintaining its original meaning and style.
<|end|>

<|user|>
Translate the following text into {{language}}: {{text}}
<|end|>
""")

if __name__ == "__main__":

    parser = TextOutputParser()
    provider = OpenAI(model="gpt-4o-mini", cache=SQLiteCache(), store=DuckDBStore(), query_per_second=0.2)
    inputs = [
        Input(text="Hello, how are you?", language="Chinese"),
        Input(text="Hello, how are you?", language="English"),
        Input(text="Hello, how are you?", language="Chinese"),
    ]

    messages = [prompt.parse(input) for input in inputs]
    responses = provider.batch(messages, batch_size=2, enable_retry=True)

    # 处理结果
    for i, response in enumerate(responses):
        print(f"\n--- Result {i+1} ---")
        print(f"Original: {inputs[i].text}")
        result = parser.parse(response)
        print(f"Translated: {result}")
        print(f"Cache Key: {response.cache_key}")
