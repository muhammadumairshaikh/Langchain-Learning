from components.config_loader import load_config
from components.retriever import build_retriever
from components.summarizer import build_summarizer


def main():
    config = load_config()

    file_path = "task_3/ai_intro.txt"
    query = "AI milestones"

    # Build retriever
    retriever = build_retriever(
    embedding_deployment=config["embedding_deployment"],
    file_path=file_path
    )   

    # Retrieve top chunks
    retrieved_docs = retriever.get_relevant_documents(query)
    # print(f"\n🔹 Retrieved {len(retrieved_docs)} relevant chunks for query: '{query}'\n")

    for i, doc in enumerate(retrieved_docs[:3], 1):  # show only top 3
        print(f"--- Chunk {i} ---")
        print(doc.page_content.strip())
        print()

    # Build summarizer (2 sentences for concise summary)
    summarizer = build_summarizer(config["chat_deployment"], sentences=2)

    # Combine retrieved docs into one string for summarization
    combined_text = " ".join(doc.page_content for doc in retrieved_docs)
    summary = summarizer.invoke({"text": combined_text})

    print("📝 Final Summary:")
    print(summary.content)


if __name__ == "__main__":
    main()
