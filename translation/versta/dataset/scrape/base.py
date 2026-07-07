from abc import ABC, abstractmethod
from pathlib import Path


class BaseScraper(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def scrape(
        self,
        source_lang: str,
        target_lang: str,
        categories: list[str],
        output_dir: Path,
        max_articles: int = 500,
        max_depth: int = 2,
    ) -> list[Path]:
        """Scrape text data and write intermediate JSONL files.

        Args:
            source_lang: Source language code.
            target_lang: Target language code.
            categories: Category names to scrape.
            output_dir: Directory to write intermediate files into.
            max_articles: Max articles per category (default: 500).
            max_depth: Max subcategory recursion depth (default: 2).

        Returns:
            List of paths to the written intermediate JSONL files.
        """
        ...
