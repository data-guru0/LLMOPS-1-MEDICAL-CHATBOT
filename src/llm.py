from langchain_huggingface import HuggingFaceEndpoint
from config.config import HUGGINGFACE_REPO_ID, get_hf_token
from src.logger import get_logger
import os
from src.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID):
    try:
        token = get_hf_token()
        if not token:
            raise CustomException("HF_TOKEN is missing.")

        logger.info(f"HF_TOKEN length: {len(token)}")

        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            task="text-generation",
            temperature=0.5,
            model_kwargs={"max_length": 512},
            huggingfacehub_api_token=token
        )

        return llm

    except Exception as e:
        logger.exception("Failed to load Hugging Face model.")
        return None
