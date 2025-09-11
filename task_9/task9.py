# task_9/task9.py

from components.config_loader import load_config
from components.retriever import build_retriever
from components.summarizer import build_summarizer


def main():
    config = load_config()
    file_path = "task_3/ai_intro.txt"
    query = "AI advancements"

    # --- 1. Normal Retriever ---
    print("\n=== Normal Retriever ===")
    retriever_normal = build_retriever(
        file_path=file_path,
        embedding_deployment=config["embedding_deployment"],
    )

    normal_docs = retriever_normal.invoke(query)
    print(f"[Normal] Retrieved {len(normal_docs)} docs")

    # extract only text
    normal_text = "\n".join([doc.page_content for doc in normal_docs])
    print("\n--- Retrieved Text (Normal) ---")
    print(normal_text)

    summarizer = build_summarizer(config["chat_deployment"], sentences=1)
    
    summary_normal = summarizer.invoke({"text": normal_text})
    print("\n--- Summary (Normal) ---")
    print(summary_normal.content if hasattr(summary_normal, "content") else summary_normal)

    # --- 2. Multi-Query Retriever ---
    print("\n=== Multi-Query Retriever ===")
    retriever_multi = build_retriever(
        file_path=file_path,
        embedding_deployment=config["embedding_deployment"],
        use_multi_query=True,   # enable multi-query
        chat_deployment=config["chat_deployment"],
    )

    multi_docs = retriever_multi.invoke(query)
    print(f"[Multi] Retrieved {len(multi_docs)} unique docs")

    # extract only text
    multi_text = "\n".join([doc.page_content for doc in multi_docs])
    print("\n--- Retrieved Text (Multi-Query) ---")
    print(multi_text)

    summary_multi = summarizer.invoke({"text": multi_text})
    print("\n--- Summary (Multi-Query) ---")
    print(summary_multi.content if hasattr(summary_multi, "content") else summary_multi)


if __name__ == "__main__":
    main()