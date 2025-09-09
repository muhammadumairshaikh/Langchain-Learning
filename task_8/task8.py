# Task 8 Script

from components.summarizer import build_summarizer
from components.config_loader import load_config


def main():
    # Load config
    config = load_config()

    # Build summarizer with StructuredOutputParser
    summarizer = build_summarizer(
        deployment_name=config["chat_deployment"], 
        sentences=3,
        use_parser=True, # JSON output
    )

    text = """
    Artificial intelligence (AI) has become one of the most transformative 
    technologies of the 21st century. Its applications span across industries, 
    reshaping the way humans work, communicate, and solve problems. In healthcare, 
    AI assists doctors by analyzing medical images, predicting disease risks, and 
    even supporting the discovery of new drugs. In transportation, self-driving 
    vehicles powered by AI promise to reduce accidents and improve traffic 
    efficiency. Businesses use AI-driven chatbots and virtual assistants to provide 
    customer service at scale, while AI-powered analytics help managers make more 
    data-informed decisions. Education also benefits from AI, with personalized 
    learning systems that adapt to each student’s pace and style. Despite its 
    benefits, AI raises concerns regarding privacy, bias, and job displacement, 
    which makes responsible AI governance crucial. As adoption grows, society must 
    balance innovation with ethical considerations to ensure AI serves humanity 
    positively.
    """

    # Summarizer
    result = summarizer.invoke({"text": text})
    
    print("=== JSON result ===")
    print(result)  # Should be dict with "summary" + "length"


if __name__ == "__main__":
    main()
