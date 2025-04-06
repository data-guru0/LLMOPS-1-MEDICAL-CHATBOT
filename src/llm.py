import logging
from langchain.llms import HuggingFaceHub
from config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    """
    Load the LLM model from Hugging Face Endpoint.

    :param huggingface_repo_id: The Hugging Face model repository ID.
    :param hf_token: The Hugging Face authentication token.
    :return: An instance of HuggingFaceEndpoint (LLM).
    """
    try:
        # Fallback: Try to re-read from system environment if hf_token is None
        if not hf_token:
            logger.warning("HF_TOKEN not passed explicitly, trying to fetch from environment again...")
            import os
            hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise ValueError("🤒 HF_TOKEN is missing. Cannot load LLM.")

        logger.info(f"✅ Loading LLM from Hugging Face: {huggingface_repo_id}")

        llm = HuggingFaceHub(
                repo_id=huggingface_repo_id,  # or "tiiuae/falcon-7b-instruct"
                model_kwargs={"temperature": 0.5, "max_length": 256},
                huggingfacehub_api_token=hf_token
            )

        logger.info("🚀 LLM successfully loaded.")
        return llm

    except Exception as e:
        error_message = CustomException("❌ Failed to load LLM from Hugging Face", e)
        logger.error(str(error_message))
        return None
