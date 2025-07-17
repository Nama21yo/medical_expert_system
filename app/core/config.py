# app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file at the project root
# This line ensures that os.getenv() will work correctly everywhere after this file is imported.
load_dotenv()

class Settings:
    # --- Project Paths ---
    CHAT_HISTORY_DIR: Path = Path("chat_history")

    # --- LLM Configuration ---
    # Load the API key from the environment. If it's not set, this will be None.
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL_NAME: str = "gemini-2.0-flash"
    
    # --- Validation ---
    def __init__(self):
        self.CHAT_HISTORY_DIR.mkdir(exist_ok=True)
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file with the key.")

# Instantiate the settings so they are validated on startup
settings = Settings()
