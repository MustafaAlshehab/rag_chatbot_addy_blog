"""
Hybrid retrieval over the ChromaDB collections.

Two-stage strategy:
  1. Q/A search    -- match the query against pre-generated Q/A pairs; a
                      high-confidence hit returns a precise, verified answer.
  2. Vector search -- fall back to chunked blog content for broader coverage.
"""

from src.vector_store import get_chroma_client, get_embedding_function

QA_SIMILARITY_THRESHOLD = 0.75
DEFAULT_TOP_K = 4


class HybridRetriever:
    """Two-stage retriever: precise Q/A matches first, document chunks as fallback."""

    def __init__(self, qa_threshold: float = QA_SIMILARITY_THRESHOLD, top_k: int = DEFAULT_TOP_K):
        self.qa_threshold = qa_threshold
        self.top_k = top_k

        client = get_chroma_client()
        ef = get_embedding_function()
        self.qa_collection = client.get_collection("qa_pairs", embedding_function=ef)
        self.chunks_collection = client.get_collection("blog_chunks", embedding_function=ef)

    def search_qa(self, query: str) -> dict | None:
        """Return the best Q/A match if its similarity clears the threshold, else None."""
        results = self.qa_collection.query(query_texts=[query], n_results=1)
        if not results["documents"][0]:
            return None

        similarity = 1 - results["distances"][0][0]
        if similarity < self.qa_threshold:
            return None

        metadata = results["metadatas"][0][0]
        return {
            "question": results["documents"][0][0],
            "answer": metadata["answer"],
            "source_page": metadata["source_page"],
            "similarity": similarity,
        }

    def search_documents(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k most similar blog chunks for the query."""
        n_results = top_k or self.top_k
        results = self.chunks_collection.query(query_texts=[query], n_results=n_results)

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "title": meta["title"],
                "source_url": meta["source_url"],
                "similarity": 1 - dist,
            })
        return chunks

    def hybrid_retrieve(self, query: str) -> dict:
        """
        Run the two-stage retrieval and return formatted context plus source info.

        Returns a dict with keys: strategy ("qa" | "vector"), context (str),
        and sources (list of {title, url}).
        """
        qa_match = self.search_qa(query)
        if qa_match is not None:
            return {
                "strategy": "qa",
                "context": _format_qa_context(qa_match),
                "sources": [{"title": "Matched Q/A pair", "url": qa_match["source_page"]}],
            }

        chunks = self.search_documents(query)
        return {
            "strategy": "vector",
            "context": _format_chunk_context(chunks),
            "sources": _unique_sources(chunks),
        }


def _format_qa_context(qa_match: dict) -> str:
    """Render a Q/A match as a single grounded context block."""
    return (
        "A verified answer to a closely matching question:\n\n"
        f"Q: {qa_match['question']}\n"
        f"A: {qa_match['answer']}"
    )


def _format_chunk_context(chunks: list[dict]) -> str:
    """Render document chunks as a numbered list of excerpts."""
    blocks = [
        f'[{i}] From "{chunk["title"]}":\n{chunk["text"]}'
        for i, chunk in enumerate(chunks, 1)
    ]
    return "Relevant excerpts from Addy Osmani's blog:\n\n" + "\n\n".join(blocks)


def _unique_sources(chunks: list[dict]) -> list[dict]:
    """Collapse chunks to unique source pages, preserving rank order."""
    seen = set()
    sources = []
    for chunk in chunks:
        url = chunk["source_url"]
        if url in seen:
            continue
        seen.add(url)
        sources.append({"title": chunk["title"], "url": url})
    return sources


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is cognitive surrender?"
    retriever = HybridRetriever()
    result = retriever.hybrid_retrieve(query)

    print(f"Query: {query!r}")
    print(f"Strategy: {result['strategy']}\n")
    print(result["context"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  - {source['title']} ({source['url']})")
