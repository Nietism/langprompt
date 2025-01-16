from pydantic import BaseModel
from langprompt.prompt import Prompt
from langprompt.output_parser import TextOutputParser
from langprompt.llms.openai import BaseLLM, OpenAI


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
Translate the following text into {{ input.language }}: {{ input.text }}
<|end|>
""",
            output_parser=TextOutputParser(),
        )

    def __call__(self, input: TranslationInput, **kwargs) -> str:
        messages = self.prompt.parse(input)
        response = self.provider.chat(messages, **kwargs)
        return self.prompt.parse_output(response)


if __name__ == "__main__":
    provider = OpenAI(model="gpt-4o-mini")

    translate = Translation(provider)
    result = translate(TranslationInput(text="Hello, how are you?", language="Chinese"))
    print(result)
