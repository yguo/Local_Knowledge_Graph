from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    LLM_MODEL = os.getenv('LLM_MODEL')
    LLM_API_URL = os.getenv('LLM_API_URL')
    FLASK_PORT = os.getenv('FLASK_PORT', 5100)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    