from pydantic import BaseModel

from langprompt import Prompt, TextOutputParser, Completion, BaseLLM
from langprompt.cache import SQLiteCache
from langprompt.llms.openai import OpenAI
from langprompt.store import DuckDBStore

class TranslationInput(BaseModel):
    text: str
    language: str = "Chinese"


class Translation:
    def __init__(self, provider: BaseLLM):
        self.provider = provider
        self.prompt = Prompt[TranslationInput, str](
            template="""
<|system|>
You are a professional translator. Please accurately translate the text while maintaining its original meaning and style.
<|end|>

<|user|>
Translate the following text into {{input.language}}: {{input.text}}
<|end|>
""",
            output_parser=TextOutputParser(),
        )

    def __call__(
        self, inputs: list[TranslationInput], batch_size: int = 2, **kwargs
    ) -> list[Completion]:
        messages = [self.prompt.parse(input) for input in inputs]
        responses = self.provider.batch(messages, batch_size=batch_size, **kwargs)
        return [response for response in responses]


if __name__ == "__main__":
    provider = OpenAI(
        model="gpt-4o-mini",
        cache=SQLiteCache(),
        store=DuckDBStore(),
        query_per_second=0.2,
    )

    translate = Translation(provider)
    inputs = [
        TranslationInput(text="Hello, how are you?", language="Chinese"),
        TranslationInput(text="Hello, how are you?", language="English"),
        TranslationInput(text="Hello, how are you?", language="Chinese"),
    ]

    results = translate(inputs, batch_size=2, enable_retry=True)

    # Process results
    for i, result in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Original: {inputs[i].text}")
        print(f"Translated: {result}")
