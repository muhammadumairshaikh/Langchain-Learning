# task-3/task3.py

# task-3/task3.py

from components.config_loader import load_config
from components.summarizer import build_summarizer
from components.retriever import build_retriever

def main():
    # Load environment variables
    config = load_config()

    # Build retriever on ai_intro.txt
    retriever = build_retriever(
        file_path="task_3/ai_intro.txt",
        env=config
    )

    # Query retriever
    query = "AI milestones"
    retrieved_docs = retriever.invoke(query)

    # Show each retrieved chunk separately
    print("\n📌 Retrieved Chunks:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content)

    # Combine retrieved text
    combined_text = " ".join([doc.page_content for doc in retrieved_docs])

    # Show complete retrieved text
    print("\n📌 Complete Retrieved Text:")
    print(combined_text)

    # Summarizer
    summarizer = build_summarizer(
        config["api_key"],
        config["endpoint"],
        config["deployment"],
        config["api_version"],
        sentences=3
    )

    # Generate summary
    summary = summarizer.invoke({"text": combined_text})

    print("\n📌 Summary:")
    print(summary["text"] if isinstance(summary, dict) else summary)


if __name__ == "__main__":
    main()
