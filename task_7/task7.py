# task_7/task7.py

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from components.config_loader import load_config
from components.summarizer import build_summarizer


def load_and_split_pdf(path: str, chunk_size: int = 150, chunk_overlap: int = 30):
    """Load and split a PDF document into chunks."""
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def load_and_split_web(url: str, chunk_size: int = 150, chunk_overlap: int = 30):
    """Load and split a webpage into chunks."""
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


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

    # --- Load Webpage ---
    web_loader = WebBaseLoader("https://builtin.com/artificial-intelligence/ai-challenges")
    web_docs = web_loader.load()

    # Clean webpage docs
    for doc in web_docs:
        doc.page_content = clean_text(doc.page_content)

    # --- Split into chunks ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    pdf_chunks = splitter.split_documents(pdf_docs)
    web_chunks = splitter.split_documents(web_docs)

    # --- Embeddings ---
    embeddings = AzureOpenAIEmbeddings(
        deployment=config["embedding_deployment"],
    )

    # --- Vector stores ---
    pdf_store = FAISS.from_documents(pdf_chunks, embeddings)
    web_store = FAISS.from_documents(web_chunks, embeddings)

    pdf_retriever = pdf_store.as_retriever()
    web_retriever = web_store.as_retriever()

    # --- Build summarizer ---
    summarizer = build_summarizer(config["chat_deployment"], sentences=3)

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