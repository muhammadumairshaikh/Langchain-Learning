# components/config_loader.py

import os
from dotenv import load_dotenv

def load_config():
    """Load environment variables and return as a dict"""
    load_dotenv()
    config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
<<<<<<< HEAD
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
=======
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
>>>>>>> tasks/task_3
    }
    return config