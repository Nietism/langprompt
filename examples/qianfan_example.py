"""
pip install langprompt[qianfan]
"""

from pydantic import BaseModel

from langprompt import Prompt, TextOutputParser
from langprompt.llms.qianfan import Qianfan


class Input(BaseModel):
    text: str
    language: str = "Chinese"


class TranslationPrompt(Prompt[Input, str]):
    def __init__(self):
        super().__init__(
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


if __name__ == "__main__":
    provider = Qianfan(model="ERNIE-4.0-Turbo-8K")

    prompt = TranslationPrompt()
    messages = prompt.parse(Input(text="Hello, how are you?", language="Chinese"))
    print(f"Messages: {messages}")
    response = provider.chat(messages)
    print(f"Response: {response}")
    result = prompt.parse_output(response)
    print(f"Result: {result}")

    print("Start to check stream")
    for chunk in prompt.stream_parse_output(provider.stream(messages)):
        print(chunk)
