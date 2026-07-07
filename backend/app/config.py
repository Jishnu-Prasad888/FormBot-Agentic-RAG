from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_provider: str = "ollama"
    llm_provider: str = "openai"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "banking-assistant"
    eval_accuracy_provider: str = "ollama"
    eval_accuracy_model: str = "banking-assistant"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    ocr_provider: str = "gemini"  # gemini | openai | ollama
    ocr_openai_model: str = "gpt-4o-mini"
    ocr_ollama_model: str = "llama3.2-vision"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
