# components/config_loader.py

# not returning api_key, endpoint, api_version
import os
from dotenv import load_dotenv

def load_config():
    """Load environment variables once and return config dict."""
    load_dotenv()
    config = {
        "chat_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    }
    return config