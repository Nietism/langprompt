#! /usr/bin/env python3

from pydantic import BaseModel
from typing import Iterator
from langprompt import TextOutputParser, Prompt
from langprompt.llms.openai import OpenAI


class Translation:
    class Input(BaseModel):
        text: str
        language: str = "Chinese"

    def __init__(self, provider: OpenAI):
        self.provider = provider
        self.prompt = Prompt[self.Input]("""
<|system|>
You are a professional translator. Please accurately translate the text while maintaining its original meaning and style.
<|end|>

<|user|>
Translate the following text into {{language}}: {{text}}
<|end|>
""")
        self.parser = TextOutputParser()

    def __call__(self, input: Input, **kwargs) -> Iterator[str]:
        messages = self.prompt.parse(input)
        response = self.provider.stream(messages, **kwargs)
        return self.parser.stream_parse(response)


if __name__ == "__main__":
    provider = OpenAI(model="gpt-4o-mini")

    translate = Translation(provider)
    result = translate(Translation.Input(text="Hello, how are you?", language="Chinese"))
    print("Output is divided by '|':")
    for chunk in result:
        print(chunk, end="|", flush=True)
