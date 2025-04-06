import logging
from langchain_huggingface import HuggingFaceEndpoint
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
        logger.info(f"Loading LLM from Hugging Face: {huggingface_repo_id}")

        # Initialize the Hugging Face LLM endpoint
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            task="text-generation",
            temperature=0.5,
            model_kwargs={
                "token": hf_token,
                "max_length": 512
            }
        )

        logger.info("LLM successfully loaded.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load LLM from Hugging Face", e)
        logger.error(str(error_message))
        return None
