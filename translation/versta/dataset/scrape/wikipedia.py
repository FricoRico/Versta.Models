import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from ..config import wikipedia_access_token
from .base import BaseScraper
from .registry import register

WIKIPEDIA_API = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "Versta-Models/1.0 (Dataset scraper; <https://github.com/versta/models>)"
API_DELAY = 0.1
MAX_CHUNK_CHARS = 2000
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(\[«])')
MAX_RETRIES = 3


@register("wikipedia")
class WikipediaCategoryScraper(BaseScraper):
    @property
    def name(self) -> str:
        return "wikipedia"

    def scrape(
        self,
        source_lang: str,
        target_lang: str,
        categories: list[str],
        output_dir: Path,
        max_articles: int = 50000,
        max_depth: int = 2,
    ) -> list[Path]:
        return self._scrape_categories(
            lang=source_lang,
            categories=categories,
            output_dir=output_dir,
            max_articles=max_articles,
            max_depth=max_depth,
        )

    def _scrape_categories(
        self,
        lang: str,
        categories: list[str],
        output_dir: Path,
        max_articles: int,
        max_depth: int = 2,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_files: list[Path] = []

        for category in categories:
            cat_slug = (
                re.sub(r"[^\w-]", "_", category).strip("_").lower() or "category"
            )
            cat_dir = output_dir / cat_slug
            cat_dir.mkdir(parents=True, exist_ok=True)
            article_count = 0

            articles = self._iter_category_members(
                lang, category, max_articles, max_depth
            )
            for title, extract in tqdm(
                articles,
                desc=f"Scraping {category}",
                unit="articles",
                total=max_articles,
                leave=False,
            ):
                article_slug = re.sub(r"[^\w-]", "_", title).strip("_").lower() or "article"
                out_path = cat_dir / f"{article_slug}.jsonl"
                sentences = _split_sentences(extract)
                with out_path.open("w", encoding="utf-8") as f:
                    for sent_idx, sent in enumerate(sentences):
                        f.write(
                            json.dumps(
                                {
                                    "text": sent,
                                    "language": lang,
                                    "category": category,
                                    "title": title,
                                    "sentence_index": sent_idx,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                all_files.append(out_path)
                article_count += 1

            if article_count > 0:
                print(f"  [{lang}] {category}: {article_count} articles")
            else:
                cat_dir.rmdir()
                print(f"  [{lang}] {category}: no articles found")

        return all_files

    def _iter_category_members(
        self,
        lang: str,
        category: str,
        max_articles: int,
        max_depth: int = 2,
    ):
        from collections import deque

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        if ":" not in category:
            category = f"Category:{category}"
        queue.append((category, 1))

        headers = {"User-Agent": USER_AGENT}
        token = wikipedia_access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        count = 0

        while queue and count < max_articles:
            cat, depth = queue.popleft()
            cat_lower = cat.lower()
            if cat_lower in visited:
                continue
            visited.add(cat_lower)

            url = WIKIPEDIA_API.format(lang=lang)
            params: dict = {
                "action": "query",
                "format": "json",
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "page|subcat",
                "prop": "extracts",
                "explaintext": True,
                "gcmlimit": 500,
            }

            for attempt in range(MAX_RETRIES):
                try:
                    while True:
                        resp = requests.get(
                            url, params=params, headers=headers, timeout=30
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        time.sleep(API_DELAY)

                        for page_id, page in (
                            data.get("query", {}).get("pages", {}).items()
                        ):
                            if page_id == "-1":
                                continue
                            ns = page.get("ns", 0)
                            title = page.get("title", "")
                            if ns == 0:
                                extract = page.get("extract", "").strip()
                                if extract:
                                    yield title, extract
                                    count += 1
                                    if count >= max_articles:
                                        return
                            elif ns == 14:
                                if depth < max_depth and title.lower() not in visited:
                                    queue.append((title, depth + 1))

                        cont = data.get("continue", {})
                        gcmcontinue = cont.get("gcmcontinue")
                        if gcmcontinue:
                            params["gcmcontinue"] = gcmcontinue
                        else:
                            break
                    break
                except (requests.RequestException, json.JSONDecodeError) as e:
                    is_429 = (
                        isinstance(e, requests.HTTPError)
                        and e.response is not None
                        and e.response.status_code == 429
                    )
                    if is_429:
                        backoff = 10 * (2**attempt)
                        print(
                            f"  Rate limited for '{cat}' ({lang}), waiting {backoff}s"
                        )
                    else:
                        backoff = 2**attempt
                        if attempt < MAX_RETRIES - 1:
                            print(f"  Retrying '{cat}' ({lang}) after {backoff}s: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(backoff)
                    else:
                        print(f"  Error fetching '{cat}' ({lang}): {e}")

            time.sleep(API_DELAY)


def _split_sentences(text: str) -> list[str]:
    sentences = SENTENCE_SPLIT.split(text)
    result: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > MAX_CHUNK_CHARS:
            for i in range(0, len(sent), MAX_CHUNK_CHARS):
                result.append(sent[i : i + MAX_CHUNK_CHARS])
        else:
            result.append(sent)
    return result or [text[:MAX_CHUNK_CHARS]]
