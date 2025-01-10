from pydantic import BaseModel
from langprompt import TextOutputParser, Prompt
from langprompt.llms.openai import BaseLLM, OpenAI


class Translation:
    class Input(BaseModel):
        text: str
        language: str = "Chinese"

    def __init__(self, provider: BaseLLM):
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

    def __call__(self, input: Input, **kwargs) -> str:
        messages = self.prompt.parse(input)
        response = self.provider.chat(messages, **kwargs)
        return self.parser.parse(response)

if __name__ == "__main__":
    provider = OpenAI(model="gpt-4o-mini")

    translate = Translation(provider)
    result = translate(Translation.Input(text="Hello, how are you?", language="Chinese"))
    print(result)
