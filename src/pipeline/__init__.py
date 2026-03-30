"""Data-ingestion and preprocessing pipeline."""

from .orchestrator import run_pipeline
from .news_scraper import NewsScraper

__all__ = ["run_pipeline", "NewsScraper"]
