from components.config_loader import load_config
from components.summarizer import build_summarizer


def main():
    config = load_config()

    text = """
    Artificial intelligence (AI) has rapidly evolved over the past few decades, transforming industries
    and reshaping the way humans interact with technology. Originally rooted in academic research
    focused on problem-solving and symbolic methods, AI has now become a cornerstone of modern
    computing, driven by machine learning and deep neural networks. In healthcare, AI systems assist
    doctors in diagnosing diseases more accurately and developing personalized treatment plans. In
    finance, algorithms detect fraudulent transactions and automate trading at speeds beyond human
    capability. The transportation sector is witnessing breakthroughs through self-driving cars and
    intelligent traffic management systems. Meanwhile, AI-powered chatbots and virtual assistants are
    revolutionizing customer service by providing instant, around-the-clock support. Despite these
    advancements, challenges persist, including bias in AI models, ethical concerns about data usage,
    and fears of job displacement due to automation. Policymakers, researchers, and industry leaders
    are increasingly collaborating to ensure that AI is developed and deployed responsibly. As the field
    continues to expand, AI holds the potential to tackle global issues such as climate change,
    healthcare accessibility, and sustainable economic growth, making it one of the most important
    technologies of the 21st century.
    """

    # 3 sentence summary
    summarizer_3 = build_summarizer(
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        deployment=config["deployment"],
        api_version=config["api_version"],
        sentences=3
    )
    result_3 = summarizer_3.run(text)
    print("\n=== 3 Sentence Summary ===")
    print(result_3)

    # 1 sentence summary
    summarizer_1 = build_summarizer(
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        deployment=config["deployment"],
        api_version=config["api_version"],
        sentences=1
    )
    result_1 = summarizer_1.run(text)
    print("\n=== 1 Sentence Summary ===")
    print(result_1)


if __name__ == "__main__":
    main()
