from langprompt import Prompt, JSONOutputParser
from langprompt.llms.openai import OpenAI
from pydantic import BaseModel


class TranslationInput(BaseModel):
    text: str
    language: str = "Chinese"

class TranslationOutput(BaseModel):
    class Result(BaseModel):
        translated: str

    result: "TranslationOutput.Result"


class TranslationPrompt(Prompt[TranslationInput, TranslationOutput]):
    def __init__(self):
        super().__init__("""
<|system|>
You are a professional translator. Please accurately translate the text while maintaining its original meaning and style.
Output should be in JSON format, example:
{"result": {"translated": "<translated text>"}}
<|end|>

<|user|>
Translate the following text into {{input.language}}: {{input.text}}
<|end|>
""",
            output_parser=JSONOutputParser(TranslationOutput),
        )


if __name__ == "__main__":
    provider = OpenAI(model="gpt-4o-mini")

    translate = TranslationPrompt()
    messages = translate.parse(TranslationInput(text="Hello, how are you?", language="Chinese"))
    response = provider.chat(messages)
    result = translate.parse_output(response)
    print(result)
