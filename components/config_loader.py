# components/config_loader.py

# not returning api_key, endpoint, api_version
import os
from dotenv import load_dotenv

def load_config():
    """Load environment variables once and return config dict."""
    load_dotenv()
    config = {
<<<<<<< HEAD
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
=======
        "chat_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
>>>>>>> tasks/task-1
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
    }
    return config