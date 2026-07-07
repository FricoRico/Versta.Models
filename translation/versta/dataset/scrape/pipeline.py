import json
import re
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import TextIO

from tqdm import tqdm

from ..llm import translate_texts
from .base import BaseScraper
from .types import ScraperEntry


def run_scrape_pipeline(
    scraper: BaseScraper,
    source_lang: str,
    target_lang: str,
    categories: list[str],
    output: Path,
    workers: int = 4,
    batch_size: int = 10,
    shard_size: int = 10000,
    max_articles: int = 50000,
    max_depth: int = 2,
    translate_only: bool = False,
) -> None:
    lang_pair = f"{source_lang}-{target_lang}"
    intermediates_dir = output / "intermediates" / scraper.name / lang_pair

    if not translate_only:
        intermediates_dir.mkdir(parents=True, exist_ok=True)

        print(f"Phase 1: Scraping categories via {scraper.name}...")
        intermediate_files = scraper.scrape(
            source_lang=source_lang,
            target_lang=target_lang,
            categories=categories,
            output_dir=intermediates_dir,
            max_articles=max_articles,
            max_depth=max_depth,
        )
    else:
        intermediate_files = sorted(intermediates_dir.rglob("*.jsonl"))
        if not intermediate_files:
            print(
                "No intermediate files found. "
                "Run without --translate-only first."
            )
            return

    output_dir = output / scraper.name / lang_pair
    output_dir.mkdir(parents=True, exist_ok=True)

    if intermediate_files:
        _translate_intermediates(
            paths=intermediate_files,
            output_dir=output_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            workers=workers,
            batch_size=batch_size,
            shard_size=shard_size,
            direction_label=f"{source_lang} \u2192 {target_lang}",
        )


def _translate_intermediates(
    paths: list[Path],
    output_dir: Path,
    source_lang: str,
    target_lang: str,
    workers: int,
    batch_size: int,
    shard_size: int,
    direction_label: str,
) -> None:
    entries: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entries.append({
                        "text": data["text"],
                        "category": data.get("category", ""),
                    })

    if not entries:
        print(f"No entries to translate for {direction_label}")
        return

    print(f"Phase 2: Translating {len(entries)} texts ({direction_label})...")

    texts = [e["text"] for e in entries]
    text_chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    entry_chunks = [entries[i : i + batch_size] for i in range(0, len(entries), batch_size)]

    stem = f"dataset_{source_lang}-{target_lang}"
    existing = sorted(output_dir.glob(f"{stem}_*.jsonl"))
    shard_index = 0
    if existing:
        match = re.search(r"_(\d+)\.jsonl$", existing[-1].name)
        if match:
            shard_index = int(match.group(1)) + 1

    pairs_in_shard = 0
    f_out: TextIO | None = None

    def _open_shard() -> None:
        nonlocal f_out, shard_index, pairs_in_shard
        if f_out is not None:
            f_out.close()
        shard_path = output_dir / f"{stem}_{shard_index:05d}.jsonl"
        f_out = shard_path.open("w", encoding="utf-8")
        shard_index += 1
        pairs_in_shard = 0

    total_written = 0

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(translate_texts, chunk, source_lang, target_lang): idx
                for idx, chunk in enumerate(text_chunks)
            }
            for future in tqdm(
                as_completed(future_map), total=len(future_map), desc="Translating"
            ):
                idx = future_map[future]
                try:
                    results = future.result()
                except Exception as e:
                    print(f"  Translation batch {idx} failed: {e}")
                    results = [None] * len(text_chunks[idx])

                for entry, translated in zip(entry_chunks[idx], results):
                    try:
                        if pairs_in_shard == 0:
                            _open_shard()

                        result = ScraperEntry(
                            source=source_lang,
                            target=target_lang,
                            input=entry["text"],
                            output=translated or "",
                            category=entry["category"],
                        )
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        pairs_in_shard += 1
                        total_written += 1

                        if pairs_in_shard >= shard_size:
                            f_out.close()
                            f_out = None
                            pairs_in_shard = 0
                    except Exception as e:
                        print(f"  Warning: failed to write entry {total_written}: {e}")
    finally:
        if f_out is not None:
            f_out.close()

    print(f"  Wrote {total_written} entries to {output_dir}/")
