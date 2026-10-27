from typing import List

import pytest

from versta import language_codes
from versta.download.download import normalize_language as download_normalize
from versta.language_codes import bundle_languages, bundle_slug


class TestNormalizeLanguageSingleOwner:
    """normalize_language lives in versta/language_codes.py; the download
    module re-export must stay the same function (NEU-133)."""

    def test_download_module_reexports_shared_function(self) -> None:
        assert download_normalize is language_codes.normalize_language


class TestBundleLanguages:
    def test_normalizes_and_sorts(self) -> None:
        assert bundle_languages(["zh-Hans", "en"]) == ["en", "zh"]

    def test_deduplicates_mirrored_pairs(self) -> None:
        assert bundle_languages(["en", "zh", "zh", "en"]) == ["en", "zh"]

    def test_keeps_traditional_chinese_distinct(self) -> None:
        assert bundle_languages(["zh-TW", "en"]) == ["en", "zh_hant"]


class TestBundleSlug:
    @pytest.mark.parametrize(
        ("codes", "expected"),
        [
            (["en", "es"], "en-es"),
            (["es", "en"], "en-es"),
            (["zh-Hans", "en"], "en-zh"),
            (["en", "zh-Hant"], "en-zh_hant"),
            (["lv", "en"], "en-lv"),
        ],
    )
    def test_slug_is_normalized_and_deterministic(
        self, codes: List[str], expected: str
    ) -> None:
        assert bundle_slug(codes) == expected

    def test_slug_is_stable_under_code_order_and_case(self) -> None:
        assert bundle_slug(["ZH-Hans", "en"]) == bundle_slug(["en", "zh"])
