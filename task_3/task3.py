from components.config_loader import load_config
from components.retriever import build_retriever
from components.summarizer import build_summarizer


def main():
    config = load_config()

    file_path = "task_3/ai_intro.txt"

    # Build retriever
    retriever = build_retriever(
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        deployment=config["embedding_deployment"],  # Embedding Model
        api_version=config["api_version"],
        file_path=file_path
    )

    # Count total chunks
    total_chunks = len(retriever.vectorstore.docstore._dict)
    print(f"\n✅ Total chunks created: {total_chunks}")

    # Query retriever
    query = "AI milestones"
    retrieved_docs = retriever.invoke(query)   

    # Show top retrieved chunks
    print("\n🔎 Top Retrieved Chunks:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:300], "...")  # show first 300 chars

    # Combine retrieved chunks into text
    retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

    # Build summarizer
    summarizer = build_summarizer(
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        deployment=config["deployment"],   # Chat model deployment
        api_version=config["api_version"],
        sentences=3
    )

    # Run summary
    summary = summarizer.run(retrieved_text)

    print("\n📝 Final Summary:")
    print(summary)


if __name__ == "__main__":
    main()
