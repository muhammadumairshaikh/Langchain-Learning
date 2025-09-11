# task_10/task10.py  (only the QA parts shown/changed)

from langchain_community.document_loaders import TextLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  
from components.config_loader import load_config
from components.summarizer import build_summarizer
from components.qa_chain import build_qa_chain

def main():
    config = load_config()

    # Load document
    loader = TextLoader("task_3/ai_intro.txt", encoding="utf-8")
    docs = loader.load()
    full_text = " ".join([d.page_content for d in docs])

    # Summarize
    summarizer = build_summarizer(config["chat_deployment"], sentences=4)
    summary = summarizer.invoke({"text": full_text})
    summary_text = summary.content if hasattr(summary, "content") else summary

    # QA Chain
    qa_chain = build_qa_chain(config["chat_deployment"])

    # Question
    question = input("\n Enter your question: ")

    # QA for Summary 
    print("\n=== QA on Summary ===")

    stream_handler = StreamingStdOutCallbackHandler()
    answer_summary = qa_chain.invoke(
        {"context": summary_text, "question": question},
        callbacks=[stream_handler],  
    )


    text_out = answer_summary.content if hasattr(answer_summary, "content") else answer_summary
    print("\n--- Model Output (Summary) ---")
    print(text_out)

    # QA for full document
    print("\n=== QA on Full Document ===")
    answer_full = qa_chain.invoke(
        {"context": full_text, "question": question},
        callbacks=[stream_handler],
    )
    text_out_full = answer_full.content if hasattr(answer_full, "content") else answer_full
    print("\n--- Model Output (Full Document) ---")
    print(text_out_full)


if __name__ == "__main__":
    main()
