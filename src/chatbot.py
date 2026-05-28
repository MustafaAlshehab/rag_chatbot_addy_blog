"""
LLM provider abstraction.
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


def get_llm(temperature: float | None = None):
    """
    Return the configured LangChain chat model based on .env settings.

    Reads LLM_PROVIDER to decide which backend to use. Currently supports
    Ollama; more can be added by importing the corresponding
    LangChain provider and adding an elif branch.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "gemma4")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature if temperature is not None else 0.7,
        )

    if provider == "openai":
        raise NotImplementedError(
            "OpenAI provider not yet configured. "
            "Install langchain-openai and add ChatOpenAI here."
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
