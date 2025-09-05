# task_1/task1.py

from components.config_loader import load_config

if __name__ == "__main__":
    config = load_config()
    print("Azure OpenAI API Key:", config["api_key"])
    print("Azure OpenAI Endpoint:", config["endpoint"])
    print("Azure OpenAI Deployment:", config["deployment"])
    print("Azure OpenAI API Version:", config["api_version"])
