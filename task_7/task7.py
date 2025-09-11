# task_7/task7.py

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from components.config_loader import load_config
from components.summarizer import build_summarizer
from components.retriever import build_retriever


def clean_text(text: str) -> str:
    """Clean webpage text to remove extra whitespace and menu noise."""
    lines = text.splitlines()
    cleaned = [
        line.strip()
        for line in lines
        if line.strip()
        and "AI Academy" not in line
        and "Toggle" not in line
        and "edit" not in line
    ]
    return " ".join(cleaned)


def main():
    config = load_config()

    # --- Load PDF ---
    pdf_loader = PyPDFLoader("task_7/ai_ethics.pdf")
    pdf_docs = pdf_loader.load()

    # --- Load Webpage (with user-agent) ---
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    web_loader = WebBaseLoader(
        "https://builtin.com/artificial-intelligence/ai-challenges",
        requests_kwargs={"headers": headers},
    )
    web_docs = web_loader.load()

    # Clean webpage docs
    for doc in web_docs:
        doc.page_content = clean_text(doc.page_content)

    # --- Retrievers ---
    pdf_retriever = build_retriever(
        embedding_deployment=config["embedding_deployment"],
        docs=pdf_docs,
        chunk_size=150,
        chunk_overlap=30,
    )
    web_retriever = build_retriever(
        embedding_deployment=config["embedding_deployment"],
        docs=web_docs,
        chunk_size=150,
        chunk_overlap=30,
    )

    # --- Build summarizer ---
    summarizer = build_summarizer(config["chat_deployment"], sentences=1)

    # --- Query both sources ---
    query = "AI challenges"

    print("\n=== PDF Results ===")
    pdf_docs = pdf_retriever.invoke(query)
    pdf_text = " ".join([d.page_content for d in pdf_docs])
    print("\n--- Retrieved PDF Text ---")
    print(pdf_text)
    pdf_summary = summarizer.invoke({"text": pdf_text})
    print("\n--- PDF Summary ---")
    print(pdf_summary.content if hasattr(pdf_summary, "content") else pdf_summary)

    print("\n=== Web Results ===")
    web_docs = web_retriever.invoke(query)
    web_text = " ".join([d.page_content for d in web_docs])
    print("\n--- Retrieved Web Text ---")
    print(web_text)
    web_summary = summarizer.invoke({"text": web_text})
    print("\n--- Web Summary ---")
    print(web_summary.content if hasattr(web_summary, "content") else web_summary)


if __name__ == "__main__":
    main()