# task_1/task1.py

import os
from components.config_loader import load_config
from dotenv import load_dotenv


def main():
    # Load everything from .env
    load_dotenv()
    config = load_config()

    print("✅ Environment Variables Loaded:")
    print(f"API Key: {os.getenv('AZURE_OPENAI_API_KEY')}")
    print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    print(f"Chat Deployment: {config['chat_deployment']}")
    print(f"Embedding Deployment: {config['embedding_deployment']}")


if __name__ == "__main__":
    main()