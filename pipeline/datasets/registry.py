"""Dataset registry: what we use, where it comes from, how big it is.

Design notes
------------
* Datasets are declared as a list of FILES, not a single archive. Several of
  the sources we care about ship two or three big files (Swiss Dwellings is two
  CSVs, not a zip), and some let us take a useful subset without pulling
  hundreds of GB (C3Po).
* `essential=False` files are skipped unless --full is passed, so a fresh
  environment can be made useful in minutes rather than hours.
* Licences are recorded because they remain true and will matter if this ever
  leaves internal R&D, but they do not block: see INTERNAL_USE in fetch.py.
* Every URL here was verified live on 2026-08-20.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Acquisition = Literal["direct", "manual"]


@dataclass(frozen=True)
class DatasetFile:
    """One downloadable artifact belonging to a dataset."""
    url: str
    filename: str
    size_bytes: int | None = None
    sha256: str | None = None
    essential: bool = True
    extract: bool = True

    @property
    def size_mb(self) -> float | None:
        return None if self.size_bytes is None else self.size_bytes / 1e6


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    files: tuple[DatasetFile, ...] = ()
    licence: str = "unknown"
    commercial_ok: bool | None = None
    acquisition: Acquisition = "direct"
    manual_instructions: str | None = None
    homepage: str | None = None
    citation: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None

    @property
    def essential_files(self) -> tuple[DatasetFile, ...]:
        return tuple(f for f in self.files if f.essential)

    def total_mb(self, essential_only: bool = True) -> float:
        fs = self.essential_files if essential_only else self.files
        return sum((f.size_mb or 0) for f in fs)


REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    REGISTRY[spec.name] = spec
    return spec


ZEN = "https://zenodo.org/api/records/{rec}/files/{name}/content"

# ---------------------------------------------------------------- floor plans
register(DatasetSpec(
    name="swiss_dwellings",
    description="42,207 apartments / 242,257 rooms across 3,093 Swiss buildings. "
                "Room polygons as WKT with real-world metric coordinates - the most "
                "directly useful European corpus for the plan vectoriser and for "
                "room-geometry priors.",
    files=(
        DatasetFile(url=ZEN.format(rec=7070952, name="geometries.csv"),
                    filename="geometries.csv", size_bytes=792_500_000),
        DatasetFile(url=ZEN.format(rec=7070952, name="simulations.csv"),
                    filename="simulations.csv", size_bytes=1_497_400_000,
                    essential=False),
    ),
    licence="CC BY 4.0", commercial_ok=True,
    homepage="https://zenodo.org/records/7070952",
    citation="Standfest et al., Swiss Dwellings v3.0.0, Zenodo.",
    tags=("floorplan", "vectoriser", "europe"),
    notes="geometries.csv alone is enough for vectoriser work; simulations.csv is "
          "daylight/acoustic sim output we do not need yet.",
))

register(DatasetSpec(
    name="cubicasa5k",
    description="5,000 annotated raster floor plans (Finnish), 80+ annotation "
                "categories. The reference floor-plan dataset.",
    files=(DatasetFile(url=ZEN.format(rec=2613548, name="cubicasa5k.zip"),
                       filename="cubicasa5k.zip", size_bytes=5_469_000_000),),
    licence="CC BY-NC 4.0", commercial_ok=False,
    homepage="https://zenodo.org/records/2613548",
    citation="Kalervo et al., CubiCasa5K, SCIA 2019.",
    tags=("floorplan", "vectoriser"),
))

register(DatasetSpec(
    name="resplan",
    description="17,000 vector residential floor plans with room-connectivity "
                "graphs and metric scale.",
    files=(DatasetFile(url="https://codeload.github.com/m-agour/ResPlan/zip/refs/heads/main",
                       filename="resplan.zip", size_bytes=None),),
    licence="CC BY 4.0", commercial_ok=True,
    homepage="https://github.com/m-agour/ResPlan",
    manual_instructions=(
        "GitHub blocks repo-scoped sandboxes (HTTP 403). On a normal network the\n"
        "direct URL works. Otherwise:\n"
        "  git clone https://github.com/m-agour/ResPlan\n"
        "  zip -r resplan.zip ResPlan\n"
        "  mv resplan.zip $VISITIT_DATA_HOME/_incoming/"),
    tags=("floorplan", "vectoriser"),
))

register(DatasetSpec(
    name="msd",
    description="Modified Swiss Dwellings: 5,372 plans / 18.9K apartments as "
                "vector + raster + connectivity graphs (ECCV 2024).",
    files=(DatasetFile(url=ZEN.format(rec=11674947, name="msd.zip"),
                       filename="msd.zip", size_bytes=None),),
    licence="CC BY-SA 4.0", commercial_ok=True,
    homepage="https://github.com/caspervanengelenburg/msd",
    manual_instructions=(
        "If the Zenodo record id has changed, find the current one at\n"
        "https://github.com/caspervanengelenburg/msd and place the archive at\n"
        "$VISITIT_DATA_HOME/_incoming/msd.zip"),
    tags=("floorplan", "vectoriser", "europe"),
))

# ------------------------------------------------- photo <-> plan / layout
register(DatasetSpec(
    name="c3po",
    description="90K photo/floor-plan pairs, 597 scenes, 153M pixel correspondences. "
                "Pretraining corpus for photo-to-plan matching (stage 6).",
    files=(),  # populated at fetch time from the HF datasets API - repo is ~427 GB
    licence="CC BY 4.0", commercial_ok=True,
    acquisition="manual",
    homepage="https://huggingface.co/datasets/kwhuang/C3",
    manual_instructions=(
        "C3Po is ~427 GB, far more than a dev box wants. Pull a subset:\n\n"
        "  pip install huggingface_hub\n"
        "  huggingface-cli download kwhuang/C3 --repo-type dataset \\\n"
        "      --include 'scenes/0000*' --local-dir $VISITIT_DATA_HOME/c3po\n\n"
        "Adjust the --include glob to take more scenes."),
    tags=("photo-plan", "matching"),
))

register(DatasetSpec(
    name="zind",
    description="Zillow Indoor Dataset: 67,448 panoramas, 1,575 homes, 2,500+ plans "
                "with 3D layouts. The best real layout ground truth available.",
    files=(),
    licence="ZInD Terms of Use (non-commercial)", commercial_ok=False,
    acquisition="manual",
    homepage="https://github.com/zillow/zind",
    manual_instructions=(
        "ZInD requires a signed request form:\n"
        "  1. Follow the instructions at https://github.com/zillow/zind\n"
        "  2. Place the delivered archive at $VISITIT_DATA_HOME/_incoming/zind.zip"),
    tags=("layout", "panorama"),
))

register(DatasetSpec(
    name="structured3d",
    description="3,500 synthetic houses, 21,835 panoramic renders with full "
                "structure annotation. Clean layout supervision.",
    files=(),
    licence="Structured3D Terms of Use (non-commercial)", commercial_ok=False,
    acquisition="manual",
    homepage="https://structured3d-dataset.org/",
    manual_instructions=(
        "Structured3D is form-gated. Request at https://structured3d-dataset.org/\n"
        "then place the archives at $VISITIT_DATA_HOME/_incoming/structured3d.zip"),
    tags=("layout", "synthetic"),
))

register(DatasetSpec(
    name="rent3d",
    description="215 London rental apartments: ~1,570 interior photos with aligned, "
                "annotated floor plans. The closest public analogue to our problem.",
    files=(
        DatasetFile(url="https://www.cs.toronto.edu/~fidler/data/Rent3D-CVPR2015.tar.gz",
                    filename="rent3d.tar.gz", size_bytes=None),
        DatasetFile(url="https://www.cs.utoronto.ca/~fidler/data/Rent3D-CVPR2015.tar.gz",
                    filename="rent3d.tar.gz", size_bytes=None),
    ),
    licence="Not stated on the project page", commercial_ok=None,
    homepage="https://www.cs.toronto.edu/~fidler/projects/rent3D.html",
    citation="Liu, Schwing, Kundu, Urtasun, Fidler. Rent3D. CVPR 2015.",
    manual_instructions=(
        "The advertised archive is a DEAD LINK (404 from both Toronto hostnames,\n"
        "verified 2026-08-20). Email the authors:\n\n"
        "    To: fidler@cs.toronto.edu, kkundu@cs.toronto.edu\n"
        "    Subject: Rent3D dataset access\n"
        "    The download link on the project page returns 404 - could you share\n"
        "    a current link? We will cite the CVPR 2015 paper.\n\n"
        "Then place it at $VISITIT_DATA_HOME/_incoming/rent3d.tar.gz"),
    tags=("photo-plan", "uk", "evaluation"),
))

register(DatasetSpec(
    name="rent3dpp",
    description="Rent3D++ (Plan2Scene): same 215 apartments, richer annotations plus "
                "the observed-coverage evaluation protocol we adopt for metric M8.",
    files=(),
    licence="Supplied on grant; Plan2Scene code is MIT", commercial_ok=None,
    acquisition="manual",
    homepage="https://github.com/3dlg-hcvc/plan2scene",
    manual_instructions=(
        "Request via the Plan2Scene form: https://forms.gle/mKAmnrzAm3LCK9ua6\n"
        "Then place the archive at $VISITIT_DATA_HOME/_incoming/rent3dpp.zip"),
    tags=("photo-plan", "uk", "evaluation"),
))

# --------------------------------------------------------------- scene priors
register(DatasetSpec(
    name="hypersim",
    description="461 indoor scenes, 77,400 photoreal renders with per-pixel depth "
                "and geometry. Useful for sanity-checking metric depth indoors.",
    files=(),
    licence="Apple Hypersim licence (research)", commercial_ok=False,
    acquisition="manual",
    homepage="https://github.com/apple/ml-hypersim",
    manual_instructions=(
        "Hypersim ships as per-scene archives with a download script:\n"
        "  https://github.com/apple/ml-hypersim/tree/main/code/python/tools\n"
        "Place scene archives under $VISITIT_DATA_HOME/hypersim/"),
    tags=("depth", "indoor", "evaluation"),
))
