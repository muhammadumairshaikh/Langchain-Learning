# Task 10 Script

from langchain_community.document_loaders import TextLoader
from components.config_loader import load_config
from components.summarizer import build_summarizer
from components.qa_chain import build_qa_chain


def main():
    config = load_config()

    # Load document 
    loader = TextLoader("task_3/ai_intro.txt", encoding="utf-8")
    docs = loader.load()
    full_text = " ".join([d.page_content for d in docs])

    print("\n✅ Full Document Loaded")
    print(f"Characters in document: {len(full_text)}")

    # Summarize
    summarizer = build_summarizer(config["chat_deployment"], sentences=4)
    summary = summarizer.invoke({"text": full_text})
    summary_text = summary.content if hasattr(summary, "content") else summary

    print("\n=== Summary of ai_intro.txt ===")
    print(summary_text)

    # QA Chain
    qa_chain = build_qa_chain(config["chat_deployment"])

    # Question
    question = input("\n Enter your question: ")

    # QA for Summary
    print("\n=== QA on Summary ===")
    answer_summary = qa_chain.invoke({"context": summary_text, "question": question})
    print(answer_summary["text"])

    # QA for full document
    print("\n=== QA on Full Document ===")
    answer_full = qa_chain.invoke({"context": full_text, "question": question})
    print(answer_full["text"])


if __name__ == "__main__":
    main()
