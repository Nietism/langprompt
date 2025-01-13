
from langprompt.base.message import Message
from langprompt.llms.openai import OpenAI
from langprompt.store.duckdb import DuckDBStore


def main():
    # 创建 DuckDB 存储
    store = DuckDBStore.connect()

    # 初始化演示用的 LLM，直接在构造函数中配置追踪
    llm = OpenAI(
        store=store,
        temperature=0
    )

    # 发送一个简单的请求
    messages = [
        Message(role="system", content="你是一个有帮助的助手。"),
        Message(role="user", content="你好！请介绍一下自己。")
    ]

    response = llm.chat(messages)
    print("Assistant:", response.content)

    # 查看存储的记录
    print("\n存储的记录:")
    for table in store._tables:
        if store._conn is not None:
            records = store._conn.execute(f"SELECT * FROM {table}").fetchall()
            for record in records:
                for i, item in enumerate(record):
                    print(f"{i}: {item}")
                print("---")

if __name__ == "__main__":
    main()
