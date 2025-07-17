# app/core/context_manager.py
import logging
from pathlib import Path
from typing import Dict

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# This global dict will store active ConversationChain instances
PERSISTENT_CHAINS: Dict[str, ConversationChain] = {}
LLM = ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL_NAME, api_key=settings.GOOGLE_API_KEY, temperature=0.2)

def get_conversation_chain(session_id: str) -> ConversationChain:
    """
    Gets or creates a ConversationChain with persistent file-based memory
    for a given session ID.
    """
    if session_id not in PERSISTENT_CHAINS:
        history_file_path = settings.CHAT_HISTORY_DIR / f"{session_id}_messages.json"

        # Create a file-based history logger
        chat_history = FileChatMessageHistory(str(history_file_path))

        # Create a memory buffer that uses the file-based history
        memory = ConversationBufferMemory(
            memory_key="history",
            chat_memory=chat_history,
            return_messages=True
        )

        # Create the ConversationChain
        PERSISTENT_CHAINS[session_id] = ConversationChain(
            llm=LLM,
            memory=memory,
            verbose=True # Set to True to see the prompt context in your logs
        )
        logger.info(f"Initialized new persistent ConversationChain for session '{session_id}'")

    return PERSISTENT_CHAINS[session_id]
