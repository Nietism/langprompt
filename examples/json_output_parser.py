from langprompt import Prompt, JSONOutputParser
from langprompt.llms.openai import OpenAI
from pydantic import BaseModel

class TranslationJSON:
    class Input(BaseModel):
        text: str
        language: str = "Chinese"

    class Result(BaseModel):
        translated: str

    class Output(BaseModel):
        result: "TranslationJSON.Result"

    def __init__(self, provider: OpenAI):
        self.provider = provider
        self.prompt = Prompt[self.Input]("""
<|system|>
You are a professional translator. Please accurately translate the text while maintaining its original meaning and style.
Output should be in JSON format, example:
{"result": {"translated": "<translated text>"}}
<|end|>

<|user|>
Translate the following text into {{language}}: {{text}}
<|end|>
""")
        self.parser = JSONOutputParser(self.Output)

    def __call__(self, input: Input, **kwargs) -> Output:
        messages = self.prompt.parse(input)
        response = self.provider.chat(messages, **kwargs)
        return self.parser.parse(response)


if __name__ == "__main__":
    provider = OpenAI(model="gpt-4o-mini")

    translate = TranslationJSON(provider)
    result = translate(TranslationJSON.Input(text="Hello, how are you?", language="Chinese"))
    print(result)
