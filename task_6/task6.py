# task_6/task6.py

from components.config_loader import load_config
from components.summarizer import build_summarizer_with_memory


def main():
    config = load_config()

    # 100-word text about Machine Learning
    machine_learning_text = """
    Machine learning is a branch of artificial intelligence that focuses on enabling computers 
    to learn patterns from data and make predictions or decisions without being explicitly programmed. 
    It involves training algorithms on large datasets to identify trends, classify information, or 
    forecast outcomes. Common applications include spam detection, recommendation systems, fraud 
    detection, and medical diagnosis. Machine learning models can be supervised, unsupervised, or 
    reinforcement-based depending on the type of data and problem. The success of machine learning 
    largely depends on data quality and computational resources, making it a driving force behind 
    modern AI advancements.
    """

    # 100-word text about Deep Learning
    deep_learning_text = """
    Deep learning is a specialized area of machine learning that uses artificial neural networks 
    with multiple layers to process complex data. It excels at handling tasks like image recognition, 
    speech processing, and natural language understanding. By mimicking how the human brain processes 
    information, deep learning can capture intricate patterns in data, leading to breakthroughs such 
    as self-driving cars, advanced chatbots, and medical image analysis. However, deep learning 
    requires massive amounts of labeled data and high computational power. Its ability to improve 
    performance with more data makes it one of the most influential technologies in modern AI development.
    """

    # 🔹 ConversationBufferMemory
    print("\n==========================")
    print("🔹 Using ConversationBufferMemory")
    print("==========================")
    buffer_summarizer = build_summarizer_with_memory(sentences=2, memory_type="buffer")
    print("Machine Learning Summary:", buffer_summarizer.invoke({"text": machine_learning_text})["text"])
    print("Deep Learning Summary:", buffer_summarizer.invoke({"text": deep_learning_text})["text"])

    # 🔹 ConversationSummaryMemory
    print("\n==========================")
    print("🔹 Using ConversationSummaryMemory")
    print("==========================")
    summary_summarizer = build_summarizer_with_memory(sentences=2, memory_type="summary")
    print("Machine Learning Summary:", summary_summarizer.invoke({"text": machine_learning_text})["text"])
    print("Deep Learning Summary:", summary_summarizer.invoke({"text": deep_learning_text})["text"])


if __name__ == "__main__":
    main()