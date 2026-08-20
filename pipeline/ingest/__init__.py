"""Listing ingestion adapters (stage -1: acquisition).

Feeds the stage 0 manifest described in docs/ARCHITECTURE.md. Every asset
carries a `provenance` tag; see docs/DATA-SOURCES.md §3.9 for why that matters.
"""
from .models import Listing, MediaRef, CoverageStats  # noqa: F401
