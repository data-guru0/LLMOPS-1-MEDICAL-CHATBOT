from langchain_huggingface import HuggingFaceEndpoint
from config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from src.logger import get_logger
import os
from src.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = None):
    try:
        # Step 1: Try to get from parameter > env > config
        token = hf_token or os.getenv("HF_TOKEN")

        if not token:
            raise CustomException("🚨 HF_TOKEN is missing. Cannot load model.")

        logger.info("✅ HF_TOKEN loaded successfully")

        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            task="text-generation",
            temperature=0.5,
            model_kwargs={"max_length": 512},
            huggingfacehub_api_token=token
        )

        logger.info("✅ LLM loaded from HuggingFaceEndpoint.")
        return llm

    except Exception as e:
        logger.exception("💥 Failed to load LLM:")
        return None

    except Exception as e:
        error_message = CustomException("Failed to load LLM from Hugging Face", e)
        logger.error(str(error_message))
        return None
