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
    provider = Qianfan(model="ERNIE-4.0-Turbo-8K", query_per_second=0.2)

    # 准备多条输入数据
    inputs = [
        Input(text="Hello, how are you?", language="Chinese"),
        Input(text="I love programming.", language="Chinese"),
        Input(text="What a beautiful day!", language="Chinese")
    ]

    # 将所有输入转换为消息格式
    all_messages = [prompt.parse(input_data) for input_data in inputs]

    # 批量处理
    responses = provider.batch(all_messages, batch_size=3)

    # 处理结果
    for i, response in enumerate(responses):
        print(f"\n--- Result {i+1} ---")
        print(f"Original: {inputs[i].text}")
        result = parser.parse(response)
        print(f"Translated: {result}")
