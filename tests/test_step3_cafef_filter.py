"""Step 3 validation: CafeF banking relevance filtering.

Run:  python -m pytest tests/test_step3_cafef_filter.py -v
"""

from __future__ import annotations

import pytest

from src.pipeline import news_scraper as ns


class TestSymbolBrandNames:

    @pytest.mark.parametrize("symbol", ["VCB", "BID"])
    def test_brand_names_exist(self, symbol: str):
        names = ns._SYMBOL_BRAND_NAMES.get(symbol)
        assert names is not None, f"No brand names for {symbol}"
        assert len(names) >= 2, f"{symbol}: only {len(names)} brand names, expected ≥2"

    @pytest.mark.parametrize("symbol", ["VCB", "BID"])
    def test_brand_names_include_ticker(self, symbol: str):
        names = ns._SYMBOL_BRAND_NAMES[symbol]
        assert symbol in names, f"{symbol}: ticker not in brand names: {names}"


class TestCafeFRelevanceFilter:

    def test_keeps_symbol_article(self):
        articles = [
            {"title": "Vietcombank lợi nhuận quý 1 tăng mạnh", "content": "Chi tiết..."},
        ]
        kept = ns._filter_cafef_by_relevance(articles, "VCB")
        assert len(kept) == 1

    def test_keeps_sector_article(self):
        articles = [
            {"title": "Ngân hàng Nhà nước sẽ giảm lãi suất", "content": "Chi tiết..."},
        ]
        for sym in ["VCB", "BID"]:
            kept = ns._filter_cafef_by_relevance(articles, sym)
            assert len(kept) == 1, f"Sector article should be kept for {sym}"

    def test_drops_unrelated_article(self):
        articles = [
            {"title": "FPT lợi nhuận tăng 30% trong quý 2", "content": "FPT Software..."},
        ]
        kept = ns._filter_cafef_by_relevance(articles, "VCB")
        assert len(kept) == 0, "Unrelated FPT article should be dropped for VCB"

    def test_mixed_batch(self):
        articles = [
            {"title": "BIDV mở rộng chi nhánh", "content": "BIDV..."},
            {"title": "FPT ra mắt sản phẩm mới", "content": "FPT..."},
            {"title": "Tỷ giá ngoại tệ biến động", "content": "USD/VND..."},
            {"title": "Vingroup mua đất", "content": "Vingroup..."},
        ]
        kept = ns._filter_cafef_by_relevance(articles, "BID")
        # BIDV article (brand match) + tỷ giá article (sector match) = 2
        assert len(kept) == 2

    def test_case_insensitive(self):
        articles = [
            {"title": "VIETCOMBANK công bố kết quả", "content": ""},
        ]
        kept = ns._filter_cafef_by_relevance(articles, "VCB")
        assert len(kept) == 1, "Brand matching should be case-insensitive"
