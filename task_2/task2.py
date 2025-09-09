from components.config_loader import load_config
from components.summarizer import build_summarizer


def main():
    config = load_config()

    input_text = (
        "Artificial Intelligence (AI) refers to the simulation of human intelligence "
        "in machines that are programmed to think like humans and mimic their actions. "
        "The term may also be applied to any machine that exhibits traits associated "
        "with a human mind such as learning and problem-solving. AI has become an "
        "essential part of the technology industry, helping to solve many challenging "
        "problems in computer science. Over the years, AI research has explored "
        "various subfields including natural language processing, robotics, machine "
        "learning, and deep learning. These technologies have led to advancements in "
        "voice assistants, autonomous vehicles, recommendation systems, and more. "
        "Despite its success, AI also raises ethical concerns, such as bias in "
        "decision-making systems, job displacement, and privacy issues. Governments, "
        "researchers, and organizations are working together to establish guidelines "
        "to ensure AI is developed and used responsibly."
    )

    # Summarizer with 3 sentences
    summarizer_3 = build_summarizer(config["chat_deployment"], sentences=3)
    print("🔹 3-sentence summary:")
    print(summarizer_3.invoke({"text": input_text}).content)

    # Summarizer with 1 sentence
    summarizer_1 = build_summarizer(config["chat_deployment"], sentences=1)
    print("\n🔹 1-sentence summary:")
    print(summarizer_1.invoke({"text": input_text}).content)


if __name__ == "__main__":
    main()
