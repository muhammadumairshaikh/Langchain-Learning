# task-2/task2.py

import os
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from components.config_loader import load_config
from components.summarizer import build_summarizer

# Load env vars
load_dotenv()
config = load_config()

# Input text
input_text = """
Artificial intelligence (AI) is a rapidly advancing field of computer science that focuses on the creation of systems capable of performing tasks that normally require human intelligence. These tasks include problem-solving, natural language understanding, perception, and decision-making. The progress in AI has been driven by advances in machine learning, particularly deep learning, which leverages large datasets and powerful computing resources to train complex models. One of the most notable applications of AI is in natural language processing, where models such as chatbots and virtual assistants are capable of understanding and responding to human language with increasing sophistication. AI also plays a significant role in computer vision, enabling systems to recognize and categorize images, detect objects, and even interpret medical scans. Additionally, AI has been integrated into recommendation engines, predictive analytics, robotics, and autonomous vehicles, transforming multiple industries. Despite its benefits, AI also raises ethical concerns, including bias in algorithms, job displacement, and questions about accountability. Governments, researchers, and organizations are actively exploring frameworks to ensure AI is developed responsibly. The potential of AI is vast, and while it offers opportunities for innovation and efficiency, it also requires careful oversight to balance progress with societal impact.
"""

# Build summarizers
summarizer_3 = build_summarizer(
    config["api_key"],
    config["endpoint"],
    config["deployment"],
    config["api_version"],
    sentences=3
)

summarizer_1 = build_summarizer(
    config["api_key"],
    config["endpoint"],
    config["deployment"],
    config["api_version"],
    sentences=1
)

# Run summarizations
summary_3 = summarizer_3.run({"text": input_text})
print("3-sentence summary:\n", summary_3)

print("------------------------------")

summary_1 = summarizer_1.run({"text": input_text})
print("\n1-sentence summary:\n", summary_1)

# Track tokens
with get_openai_callback() as cb:
    summarizer_3.run({"text": input_text})
    print("\n--- Token Usage (3-sentence) ---")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")

with get_openai_callback() as cb:
    summarizer_1.run({"text": input_text})
    print("\n--- Token Usage (1-sentence) ---")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
