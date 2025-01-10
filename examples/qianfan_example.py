"""
pip install langprompt[qianfan]
"""
from pydantic import BaseModel
from langprompt import TextOutputParser, Prompt
from langprompt.llms.qianfan import Qianfan



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

parser = TextOutputParser()


if __name__ == "__main__":
    provider = Qianfan(model="ERNIE-4.0-Turbo-8K")

    messages = prompt.parse(Input(text="Hello, how are you?", language="Chinese"))
    print(f"Messages: {messages}")
    response = provider.chat(messages)
    print(f"Response: {response}")
    result = parser.parse(response)
    print(f"Result: {result}")

    print("Start to check stream")
    for chunk in provider.stream(messages):
        print(chunk)
