"""
LLM provider abstraction and the RAG chatbot.

`RAGChatbot` wires hybrid retrieval, grounded generation, and windowed conversation memory together.
"""

import os
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.retriever import HybridRetriever

load_dotenv()

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about Addy Osmani's blog \
posts on software engineering, AI, and coding agents.

Answer using ONLY the context below. If it does not contain the answer, say so \
honestly instead of guessing. Be concise and accurate, and answer naturally \
without referring to "the context".

Context:
{context}"""


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


class RAGChatbot:
    """Hybrid retrieval + grounded generation with windowed conversation memory."""

    def __init__(self, memory_turns: int = 5, temperature: float = 0.3):
        self.retriever = HybridRetriever()
        self.llm = get_llm(temperature=temperature)
        self.memory_turns = memory_turns
        self.history: list[BaseMessage] = []

    def chat(self, query: str) -> dict:
        """
        Answer a user query grounded in retrieved context.

        Returns:
            dict with keys:
                - answer (str): the model's grounded response
                - strategy (str): either "qa" or "vector" (retrieval method used)
                - sources (list): [{title, url}] list of source documents used
        """
        retrieval = self.retriever.hybrid_retrieve(query)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=retrieval["context"])),
            *self._recent_history(),
            HumanMessage(content=query),
        ]

        response = self.llm.invoke(messages)
        answer = _strip_reasoning(response.content)

        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "strategy": retrieval["strategy"],
            "sources": retrieval["sources"],
        }

    def reset(self):
        """Clear conversation memory."""
        self.history = []

    def _recent_history(self) -> list[BaseMessage]:
        """Return the last `memory_turns` exchanges (two messages per turn)."""
        return self.history[-self.memory_turns * 2:]

def _strip_reasoning(text: str) -> str:
    """Remove <think> blocks emitted by reasoning models like qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


if __name__ == "__main__":
    bot = RAGChatbot()
    print("RAG chatbot ready. Type 'exit' to quit, 'clear' to reset memory.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "clear":
            bot.reset()
            print("(memory cleared)\n")
            continue

        result = bot.chat(query)
        print(f"\nBot [{result['strategy']}]: {result['answer']}")
        if result["sources"]:
            print("Sources:")
            for source in result["sources"]:
                print(f"  - {source['title']} ({source['url']})")
        print()
