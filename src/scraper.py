"""
Web scraper for addyosmani.com/blog/

Scrapes blog posts from Addy Osmani's personal blog.
The site is server-side rendered.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://addyosmani.com"
BLOG_URL = f"{BASE_URL}/blog/"
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGChatbot/1.0; educational project)"
}

REQUEST_DELAY = 5  # seconds between requests


def fetch_page(url: str) -> str | None:
    """Fetch a page and return its HTML content, or None on failure."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}")
        return None


def extract_article_links(html: str) -> list[str]:
    """Extract all personal blog post links from the blog listing page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for card in soup.select("article.card"):
        anchor = card.select_one("h3.card-title a")
        if anchor and anchor.get("href", "").startswith("/blog/"):
            full_url = urljoin(BASE_URL, anchor["href"])
            links.append(full_url)

    return links


def parse_article(html: str, url: str) -> dict | None:
    """Parse an article page and extract structured data."""
    soup = BeautifulSoup(html, "html.parser")

    article = soup.select_one("article.post")
    if not article:
        print(f"  [WARN] No article.post found at {url}")
        return None

    title_tag = article.select_one("header h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    date_tag = article.select_one("header h2.headline")
    date = date_tag.get_text(strip=True) if date_tag else ""

    body = article.select_one("section#post-body")
    if not body:
        print(f"  [WARN] No post-body found at {url}")
        return None

    content = clean_text(body)

    word_count = len(content.split())
    if word_count < 50:
        print(f"  [WARN] Very short content ({word_count} words) at {url}")
        return None

    return {
        "url": url,
        "title": title,
        "date": date,
        "content": content,
        "word_count": word_count,
    }


def clean_text(body_element) -> str:
    """Extract clean text from a BeautifulSoup element, removing unwanted tags."""
    for tag in body_element.find_all(["script", "style", "nav", "footer", "iframe"]):
        tag.decompose()

    for img in body_element.find_all("img"):
        img.decompose()

    text = body_element.get_text(separator="\n")

    # Collapse multiple blank lines, strip trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def save_article(article: dict) -> Path:
    """Save an article dict as JSON to data/raw/."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    slug = article["url"].rstrip("/").split("/")[-1]
    filepath = RAW_DATA_DIR / f"{slug}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(article, f, indent=2, ensure_ascii=False)

    return filepath


def scrape_single(url: str) -> dict | None:
    """Scrape a single article by URL."""
    html = fetch_page(url)
    if not html:
        return None
    return parse_article(html, url)


def collect_all_links(max_pages: int = 5) -> list[str]:
    """
    Crawl paginated listing pages to collect all article links.
    Pages follow the pattern: /blog/, /blog/page2/, /blog/page3/, ...
    Stops when a page has no articles or max_pages is reached.
    """
    all_links = []

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            url = BLOG_URL
        else:
            url = f"{BLOG_URL}page{page_num}/"

        print(f"Fetching listing page {page_num}: {url}")
        html = fetch_page(url)
        if not html:
            break

        links = extract_article_links(html)
        if not links:
            print(f"  No articles found on page {page_num}, stopping.")
            break

        print(f"  Found {len(links)} articles on page {page_num}.")
        all_links.extend(links)
        time.sleep(REQUEST_DELAY)

    return all_links


def scrape_all(max_pages: int = 5, max_articles: int = 30) -> list[dict]:
    """
    Full crawl pipeline:
    1. Fetch paginated listing pages to collect all article links
    2. Scrape each article with rate limiting
    3. Save results to data/raw/
    """
    links = collect_all_links(max_pages=max_pages)
    print(f"\nCollected {len(links)} total article links across listing pages.")

    if max_articles:
        links = links[:max_articles]
        print(f"Limiting to {max_articles} articles.\n")

    articles = []
    for i, link in enumerate(links, 1):
        print(f"[{i}/{len(links)}] Scraping: {link}")
        article = scrape_single(link)

        if article:
            filepath = save_article(article)
            articles.append(article)
            print(f"  Saved: {filepath.name} ({article['word_count']} words)")
        else:
            print(f"  Skipped.")

        if i < len(links):
            time.sleep(REQUEST_DELAY)

    print(f"\nDone! Scraped {len(articles)}/{len(links)} articles.")
    return articles


if __name__ == "__main__":
    scrape_all()
