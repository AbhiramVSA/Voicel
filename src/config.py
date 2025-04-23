from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(dotenv_path=".env", override=True)


class Settings(BaseSettings):
    GROQ_API_KEY: str 
    GEMINI_API_KEY: str 
    WHISPER_URL: str  
    SANITIZE_URL: str 
    LOGFIRE_KEY : str
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    
    #If you have validation errors, uncomment the line below and comment the 'Config' Class:
    # model_config = SettingsConfigDict(env_file=os.path.join(os.path.dirname(__file__), "..", ".env")) 
