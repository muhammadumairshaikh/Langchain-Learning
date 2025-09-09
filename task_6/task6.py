# task_6/task6.py

from components.config_loader import load_config
from components.summarizer import build_summarizer, create_buffer_memory, create_summary_memory
from langchain_openai import AzureChatOpenAI


def main():
    config = load_config()

    # Sample texts
    text_ml = (
        """
        Machine learning is a branch of artificial intelligence that focuses on enabling computers 
        to learn patterns from data and make predictions or decisions without being explicitly programmed. 
        It involves training algorithms on large datasets to identify trends, classify information, or 
        forecast outcomes. Common applications include spam detection, recommendation systems, fraud 
        detection, and medical diagnosis. Machine learning models can be supervised, unsupervised, or 
        reinforcement-based depending on the type of data and problem. The success of machine learning 
        largely depends on data quality and computational resources, making it a driving force behind 
        modern AI advancements.
        """
    )

    text_dl = (
        """
        Deep learning is a specialized area of machine learning that uses artificial neural networks 
        with multiple layers to process complex data. It excels at handling tasks like image recognition, 
        speech processing, and natural language understanding. By mimicking how the human brain processes 
        information, deep learning can capture intricate patterns in data, leading to breakthroughs such 
        as self-driving cars, advanced chatbots, and medical image analysis. However, deep learning 
        requires massive amounts of labeled data and high computational power. Its ability to improve 
        performance with more data makes it one of the most influential technologies in modern AI development.
        """
    )

    # --- 1. Using ConversationBufferMemory ---
    print("\n=== Using ConversationBufferMemory ===")
    buffer_memory = create_buffer_memory(k=3)
    summarizer_buffer = build_summarizer(
        config["chat_deployment"],
        sentences=2,
        memory=buffer_memory,
    )

    print("\n--- Summarizing Machine Learning ---")
    print(summarizer_buffer.invoke({"text": text_ml})["text"])

    print("\n--- Summarizing Deep Learning (with buffer memory) ---")
    print(summarizer_buffer.invoke({"text": text_dl})["text"])

    # --- 2. Using ConversationSummaryMemory ---
    print("\n=== Using ConversationSummaryMemory ===")
    llm = AzureChatOpenAI(deployment_name=config["chat_deployment"], temperature=0)
    summary_memory = create_summary_memory(llm)
    summarizer_summary = build_summarizer(
        config["chat_deployment"],
        sentences=2,
        memory=summary_memory,
    )

    print("\n--- Summarizing Machine Learning ---")
    print(summarizer_summary.invoke({"text": text_ml})["text"])

    print("\n--- Summarizing Deep Learning (with summary memory) ---")
    print(summarizer_summary.invoke({"text": text_dl})["text"])


if __name__ == "__main__":
    main()
