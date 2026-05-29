"""
Vector store module: document chunking, embedding, and ChromaDB ingestion.

Handles two collections:
  - "blog_chunks": chunked blog post content for broad vector search
  - "qa_pairs": Q/A pairs for precise semantic matching
"""

import json
import os
from pathlib import Path


import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
QA_DATASET_PATH = Path(__file__).parent.parent / "data" / "qa_dataset.csv"
CHROMA_DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embedding_function():
    """Return the SentenceTransformer embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_chroma_client():
    """Return a persistent ChromaDB client stored at data/chroma_db/."""
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def load_documents() -> list[dict]:
    """Load all scraped blog posts from data/raw/ as a list of dicts."""
    documents = []
    for json_file in sorted(RAW_DATA_DIR.glob("*.json")):
        with open(json_file) as f:
            doc = json.load(f)
            documents.append(doc)
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into retrieval-friendly chunks using
    RecursiveCharacterTextSplitter.

    Returns a list of dicts with keys: text, metadata (source url, title, chunk_index).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, text in enumerate(splits):
            chunks.append({
                "text": text,
                "metadata": {
                    "source_url": doc["url"],
                    "title": doc["title"],
                    "date": doc.get("date", ""),
                    "chunk_index": i,
                    "total_chunks": len(splits),
                },
            })
    return chunks


def load_qa_pairs() -> list[dict]:
    """Load Q/A pairs from the CSV dataset."""
    qa_dataframe = pd.read_csv(QA_DATASET_PATH)  # load CSV into a pandas DataFrame

    pairs = []
    for _, row in qa_dataframe.iterrows():
        pairs.append({
            "question": row["question"],
            "answer": row["answer"],
            "source_page": row["source_page"],
        })
    return pairs

def ingest_chunks(chunks: list[dict], client: Optional["chromadb.api.types.Client"] = None, reset: bool = False):
    """
    Embed and store document chunks in the 'blog_chunks' ChromaDB collection.

    Args:
        chunks: Output of chunk_documents().
        client: Optional ChromaDB client (creates one if not provided).
        reset: If True, delete and recreate the collection.
    """
    if client is None:
        client = get_chroma_client()


    ef = get_embedding_function()

    if reset:
        try:
            client.delete_collection("blog_chunks")
        except ValueError:
            pass

    collection = client.get_or_create_collection(
        name="blog_chunks",
        embedding_function=ef,
        # The metadata parameter allows you to configure the index. 
        # The key "hnsw:space": "cosine" sets the distance metric used for nearest neighbor search in ChromaDB to cosine similarity.
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not reset:
        print(f"  blog_chunks already has {collection.count()} entries, skipping.")
        return collection

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"  Ingested {collection.count()} chunks into 'blog_chunks'.")
    return collection

def ingest_qa_pairs(qa_pairs: list[dict], client: Optional["chromadb.api.types.Client"] = None, reset: bool = False):
    """
    Embed and store Q/A pairs in the 'qa_pairs' ChromaDB collection.

    Each entry is embedded by the question text, with the answer stored as metadata
    so the retriever can return it directly on a match.
    """
    if client is None:
        client = get_chroma_client()

    ef = get_embedding_function()

    if reset:
        try:
            client.delete_collection("qa_pairs")
        except ValueError:
            pass

    collection = client.get_or_create_collection(
        name="qa_pairs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not reset:
        print(f"  qa_pairs already has {collection.count()} entries, skipping.")
        return collection

    batch_size = 100
    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i : i + batch_size]
        collection.add(
            ids=[f"qa_{i + j}" for j in range(len(batch))],
            documents=[p["question"] for p in batch],
            metadatas=[
                {"answer": p["answer"], "source_page": p["source_page"]}
                for p in batch
            ],
        )

    print(f"  Ingested {collection.count()} Q/A pairs into 'qa_pairs'.")
    return collection


def build_vector_store(reset: bool = False):
    """
    Full pipeline: load data, chunk documents, embed everything into ChromaDB.

    Args:
        reset: If True, wipe existing collections and rebuild from scratch.
    """
    print("Loading documents...")
    documents = load_documents()
    print(f"  Loaded {len(documents)} blog posts.")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("Loading Q/A pairs...")
    qa_pairs = load_qa_pairs()
    print(f"  Loaded {len(qa_pairs)} Q/A pairs.")

    client = get_chroma_client()

    print("Ingesting chunks into ChromaDB...")
    ingest_chunks(chunks, client=client, reset=reset)

    print("Ingesting Q/A pairs into ChromaDB...")
    ingest_qa_pairs(qa_pairs, client=client, reset=reset)

    print("Done! Vector store is ready.")
    return client


def test_retrieval(query: str, n_results: int = 3):
    """Quick test: search both collections and print results."""
    client = get_chroma_client()
    ef = get_embedding_function()

    print(f"\nQuery: '{query}'\n")

    chunks_col = client.get_collection("blog_chunks", embedding_function=ef)
    results = chunks_col.query(query_texts=[query], n_results=n_results)

    print("--- Blog Chunks Results ---")
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        similarity = 1 - dist
        print(f"  [{i+1}] (sim={similarity:.3f}) [{meta['title']}]")
        print(f"      {doc[:120]}...")
        print()

    qa_col = client.get_collection("qa_pairs", embedding_function=ef)
    results = qa_col.query(query_texts=[query], n_results=n_results)

    print("--- Q/A Pairs Results ---")
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        similarity = 1 - dist
        print(f"  [{i+1}] (sim={similarity:.3f}) Q: {doc[:100]}...")
        print(f"      A: {meta['answer'][:120]}...")
        print()


if __name__ == "__main__":
    import sys

    if "--reset" in sys.argv:
        build_vector_store(reset=True)
    elif "--test" in sys.argv:
        query = " ".join(sys.argv[sys.argv.index("--test") + 1:]) or "What is cognitive surrender?"
        test_retrieval(query)
    else:
        build_vector_store()
