from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.llm import load_llm
from src.vector_store import load_vector_store
from config.config import HUGGINGFACE_REPO_ID
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

def create_qa_chain():
    try:
        logger.info("Loading vector store...")
        db = load_vector_store()
        if db is None:
            raise CustomException("Vector store is empty. Load PDFs and create embeddings first.")

        logger.info("Initializing LLM for RetrievalQA...")
        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID)

        if llm is None:
            raise CustomException("LLM failed to load. Check HF_TOKEN.")

        logger.info("Creating RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        logger.info("QA chain successfully created.")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Error while creating QA chain", e)
        logger.error(str(error_message))
        return None
