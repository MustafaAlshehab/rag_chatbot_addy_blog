"""
Synthetic Q/A pair generation from scraped blog content.

Loads scraped articles, sends each to the configured LLM,
parses structured Q/A output, and saves the full dataset to CSV.
"""

import csv
import json
import re
from pathlib import Path

from src.chatbot import get_llm

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
QA_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "qa_dataset.csv"

QA_PROMPT_TEMPLATE = """\
You are an expert at creating high-quality question-answer pairs from technical blog posts.

Given the following article by Addy Osmani, generate exactly {num_pairs} question-answer pairs.

Rules:
- Mix question types: factual (what/who/when), conceptual (why/how), and inferential (implications/comparisons).
- Questions must be self-contained — a reader should understand them without seeing the article.
- Answers must be 2-4 sentences, accurate, and grounded in the article text.
- Cover different sections and key ideas; avoid clustering questions around a single paragraph.
- Do NOT generate questions about the author personally unless it directly relates to the topic.

Article Title: {title}

Article Content:
{content}

Respond with ONLY a JSON array. No markdown fences, no commentary, no preamble — just the raw JSON array:
[
  {{"question": "...", "answer": "..."}},
  ...
]"""


def load_articles() -> list[dict]:
    """Load all scraped articles from data/raw/ sorted by filename."""
    articles = []
    for filepath in sorted(RAW_DATA_DIR.glob("*.json")):
        with open(filepath, encoding="utf-8") as f:
            article = json.load(f)
            article["_filename"] = filepath.stem
            articles.append(article)
    return articles


def generate_qa_pairs(article: dict, llm, num_pairs: int = 10) -> list[dict]:
    """Send a single article to the LLM and return parsed Q/A pairs."""
    content = article["content"]

    max_chars = 20_000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[Content truncated]"

    prompt = QA_PROMPT_TEMPLATE.format(
        num_pairs=num_pairs,
        title=article["title"],
        content=content,
    )

    response = llm.invoke(prompt)
    return parse_qa_response(response.content)


def parse_qa_response(text: str) -> list[dict]:
    """
    Parse LLM response text into a list of {"question", "answer"} dicts.

    Handles markdown fences, <think> blocks, and partial JSON gracefully.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip()

    for candidate in [text, _extract_json_array(text)]:
        if candidate is None:
            continue
        try:
            pairs = json.loads(candidate)
            if isinstance(pairs, list):
                return _validate_pairs(pairs)
        except (json.JSONDecodeError, TypeError):
            continue

    print("  [WARN] Failed to parse Q/A response from LLM")
    return []


def _extract_json_array(text: str) -> str | None:
    """Try to pull a JSON array out of surrounding prose."""
    match = re.search(r"\[.*]", text, re.DOTALL)
    return match.group() if match else None


def _validate_pairs(pairs: list) -> list[dict]:
    """Keep only well-formed pairs with non-empty question and answer."""
    valid = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        q = p.get("question", "").strip()
        a = p.get("answer", "").strip()
        if q and a:
            valid.append({"question": q, "answer": a})
    return valid


def save_qa_dataset(all_pairs: list[dict], output_path: Path = QA_OUTPUT_PATH):
    """Write all Q/A pairs to CSV with columns: question, answer, source_page."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "source_page"])
        writer.writeheader()
        writer.writerows(all_pairs)
    print(f"\nSaved {len(all_pairs)} Q/A pairs to {output_path}")


def generate_all(num_pairs_per_article: int = 10) -> list[dict]:
    """
    Full pipeline: load every article, generate Q/A pairs, save to CSV.

    Returns the combined list of all pairs.
    """
    articles = load_articles()
    print(f"Loaded {len(articles)} articles from {RAW_DATA_DIR}\n")

    llm = get_llm(temperature=0.7)

    all_pairs: list[dict] = []
    for i, article in enumerate(articles, 1):
        print(f"[{i}/{len(articles)}] {article['title']}")

        pairs = generate_qa_pairs(article, llm, num_pairs=num_pairs_per_article)
        for pair in pairs:
            pair["source_page"] = article["url"]

        all_pairs.extend(pairs)
        print(f"  → {len(pairs)} pairs generated")

    save_qa_dataset(all_pairs)
    return all_pairs


if __name__ == "__main__":
    generate_all()
