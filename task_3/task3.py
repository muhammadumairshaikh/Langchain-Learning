# task-3/task3.py

# task-3/task3.py

from components.config_loader import load_config
from components.summarizer import build_summarizer
from components.retriever import build_retriever

def main():
    config = load_config()

    # Build retriever
    retriever, docs = build_retriever(
        file_path="task_3/ai_intro.txt",  
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        embedding_deployment="text-embedding-3-small",  
        api_version=config["api_version"]
    )

    # Debug info: total docs and sample chunk
    print(f"\n Total chunks created: {len(docs)}")
    if docs:
        print(f"\n Sample chunk (first 200 chars):\n{docs[0].page_content[:200]}")

    # Query retriever
    query = "AI milestones"
    retrieved_docs = retriever.invoke(query)

    # Summarizer
    summarizer = build_summarizer(
        config["api_key"],
        config["endpoint"],
        config["deployment"],   
        config["api_version"],
        sentences=3
    )

    # Retrieval
    retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])
    summary = summarizer.invoke({"text": retrieved_text})

    print("\n Summary of AI milestones:\n", summary)

if __name__ == "__main__":
    main()
