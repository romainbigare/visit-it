"""Typed records shared by all ingestion adapters."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Provenance = Literal["scraped", "partner_granted", "self_captured"]


@dataclass
class MediaRef:
    """One image belonging to a listing."""
    url: str
    kind: Literal["photo", "floorplan", "epc"]
    caption: str | None = None
    local_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Listing:
    """A single property listing, normalised across portals.

    Field names deliberately mirror what stage 0 (triage) expects so the
    manifest builder does not need a translation layer.
    """
    listing_id: str
    source: Literal["rightmove", "zoopla"]
    url: str
    provenance: Provenance = "scraped"

    display_address: str | None = None
    outcode: str | None = None
    country: str | None = None
    property_subtype: str | None = None
    transaction_type: str | None = None
    tenure: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    price_gbp: int | None = None

    # The UK area problem in one field: usually absent. See docs/DATA-SOURCES.md §5.
    floor_area_sqm: float | None = None
    floor_area_source: str | None = None

    photos: list[MediaRef] = field(default_factory=list)
    floorplans: list[MediaRef] = field(default_factory=list)

    description: str | None = None
    key_features: list[str] = field(default_factory=list)
    agent: str | None = None
    fetched_at: str | None = None

    @property
    def has_floorplan(self) -> bool:
        return len(self.floorplans) > 0

    @property
    def n_photos(self) -> int:
        return len(self.photos)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["has_floorplan"] = self.has_floorplan
        d["n_photos"] = self.n_photos
        return d


@dataclass
class CoverageStats:
    """Floor-plan coverage over *everything seen*, not just what we kept.

    This is the honest denominator: selecting only listings that have plans and
    then reporting 100% coverage would be worthless. `seen` counts every search
    result inspected, before any filtering.
    """
    source: str
    seen: int = 0
    with_floorplan: int = 0
    without_floorplan: int = 0
    with_multiple_floorplans: int = 0
    with_area: int = 0
    photo_counts: list[int] = field(default_factory=list)
    detail_fetched: int = 0
    detail_failed: int = 0

    def observe(self, n_floorplans: int, n_photos: int | None = None, has_area: bool = False) -> None:
        self.seen += 1
        if n_floorplans > 0:
            self.with_floorplan += 1
            if n_floorplans > 1:
                self.with_multiple_floorplans += 1
        else:
            self.without_floorplan += 1
        if has_area:
            self.with_area += 1
        if n_photos is not None:
            self.photo_counts.append(n_photos)

    @property
    def floorplan_rate(self) -> float:
        return (self.with_floorplan / self.seen) if self.seen else 0.0

    @property
    def area_rate(self) -> float:
        return (self.with_area / self.seen) if self.seen else 0.0

    @property
    def median_photos(self) -> float | None:
        if not self.photo_counts:
            return None
        s = sorted(self.photo_counts)
        n = len(s)
        return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "listings_seen": self.seen,
            "with_floorplan": self.with_floorplan,
            "without_floorplan": self.without_floorplan,
            "floorplan_rate": round(self.floorplan_rate, 4),
            "with_multiple_floorplans": self.with_multiple_floorplans,
            "with_stated_area": self.with_area,
            "area_rate": round(self.area_rate, 4),
            "median_photos_per_listing": self.median_photos,
            "detail_pages_fetched": self.detail_fetched,
            "detail_pages_failed": self.detail_failed,
        }
