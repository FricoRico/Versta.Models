import os
from argparse import ArgumentParser
from pathlib import Path

from .pipeline import run_scrape_pipeline
from .registry import get_scraper, list_scrapers


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="Scrape Wikipedia categories and translate via LLM to generate parallel data.",
    )

    parser.add_argument(
        "--scraper",
        type=str,
        default="wikipedia",
        help=f"Scraper to use. Available: {', '.join(list_scrapers())} (default: wikipedia)",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source language code (e.g. 'en').",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target language code (e.g. 'nl').",
    )

    parser.add_argument(
        "--category",
        type=str,
        action="append",
        dest="categories",
        required=True,
        help="Wikipedia category name to scrape (repeatable, e.g. --category 'ML' --category 'DL').",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/scrape"),
        help="Path to the output folder.",
    )

    parser.add_argument(
        "--max-articles",
        type=int,
        default=50000,
        help="Maximum articles to scrape per category (default: 50000).",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum subcategory recursion depth (default: 2, 1=no subcategories).",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for LLM inference (default: 4).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Texts per LLM batch request (default: 10).",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of pairs per shard for input merging and output processing.",
    )

    parser.add_argument(
        "--translate-only",
        action="store_true",
        default=False,
        help="Skip scraping; re-translate from existing intermediate files.",
    )

    return parser.parse_args()


def main(
    scraper: str,
    source: str,
    target: str,
    categories: list[str],
    output: Path,
    workers: int,
    batch_size: int,
    shard_size: int,
    max_articles: int,
    max_depth: int,
    translate_only: bool,
) -> None:
    scraper_instance = get_scraper(scraper)

    run_scrape_pipeline(
        scraper=scraper_instance,
        source_lang=source,
        target_lang=target,
        categories=categories,
        output=output,
        workers=workers,
        batch_size=batch_size,
        shard_size=shard_size,
        max_articles=max_articles,
        max_depth=max_depth,
        translate_only=translate_only,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        scraper=args.scraper,
        source=args.source,
        target=args.target,
        categories=args.categories,
        output=args.output,
        workers=args.workers,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        max_articles=args.max_articles,
        max_depth=args.max_depth,
        translate_only=args.translate_only,
    )
