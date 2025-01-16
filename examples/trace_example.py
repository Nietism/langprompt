from langprompt.base.message import Message
from langprompt.llms.openai import OpenAI
from langprompt.store.duckdb import DuckDBStore


def main():
    # Create DuckDB storage
    store = DuckDBStore.connect()

    # Initialize demo LLM with tracing configured in constructor
    llm = OpenAI(
        store=store,
        temperature=0
    )

    # Send a simple request
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello! Please introduce yourself.")
    ]

    response = llm.chat(messages)
    print("Assistant:", response.content)

    # View stored records
    print("\nStored records:")
    for table in store._tables:
        if store._conn is not None:
            records = store._conn.execute(f"SELECT * FROM {table}").fetchall()
            for record in records:
                for i, item in enumerate(record):
                    print(f"{i}: {item}")
                print("---")

if __name__ == "__main__":
    main()
