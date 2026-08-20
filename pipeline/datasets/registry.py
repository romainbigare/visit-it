"""Dataset registry: what we may download, from where, and under what licence.

Every entry mirrors a row in docs/LICENSING.md. The `verdict` field is load
bearing: `research_only` and `licence_unclear` datasets are refused by default
so nobody trains a shipped model on encumbered data by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["prod_ok", "research_only", "licence_unclear"]
Acquisition = Literal["direct", "manual"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    urls: tuple[str, ...]
    licence: str
    verdict: Verdict
    acquisition: Acquisition = "direct"
    sha256: dict[str, str] = field(default_factory=dict)
    manual_instructions: str | None = None
    homepage: str | None = None
    citation: str | None = None
    approx_size: str | None = None
    # Files expected after extraction; used to decide "already present".
    expect: tuple[str, ...] = ()

    @property
    def needs_ack(self) -> bool:
        return self.verdict != "prod_ok"


REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    REGISTRY[spec.name] = spec
    return spec


# --------------------------------------------------------------- Rent3D
# URLs are filled in by tools/refresh_rent3d_urls.py; see docs/PRIOR-ART.md for
# why this dataset matters and docs/LICENSING.md for its (unstated) licence.
register(DatasetSpec(
    name="rent3d",
    description=("215 London rental apartments: ~1,570 interior photos with annotated "
                 "floor plans, room layouts, walls, doors and windows (CVPR 2015)."),
    # The project page advertises ../data/Rent3D-CVPR2015.tar.gz but that file
    # returns HTTP 404 as of 2026-08-20 (verified from two hostnames, and
    # independently). We keep the canonical URLs first in case it is restored,
    # then try the Internet Archive, then fall back to the email route below.
    urls=(
        "https://www.cs.toronto.edu/~fidler/data/Rent3D-CVPR2015.tar.gz",
        "https://www.cs.utoronto.ca/~fidler/data/Rent3D-CVPR2015.tar.gz",
        "https://web.archive.org/web/2020id_/http://www.cs.toronto.edu/~fidler/data/Rent3D-CVPR2015.tar.gz",
    ),
    licence="Not stated on the project page - must be confirmed with the authors",
    verdict="licence_unclear",
    manual_instructions=(
        "The Rent3D archive advertised on the project page is a dead link (404).\n"
        "Obtain it by emailing the authors:\n\n"
        "    To: fidler@cs.toronto.edu, kkundu@cs.toronto.edu\n"
        "    Subject: Rent3D dataset access + licence terms\n\n"
        "    We are building a 3D reconstruction system for UK property listings\n"
        "    and would like to use Rent3D as an evaluation set. The download link\n"
        "    on the project page (../data/Rent3D-CVPR2015.tar.gz) returns 404.\n"
        "    Could you share a current link, and confirm the licence terms -\n"
        "    specifically whether evaluation use by a commercial entity is\n"
        "    permitted? We will cite the CVPR 2015 paper in any published work.\n\n"
        "When you receive it, place the archive at:\n"
        "    $VISITIT_DATA_HOME/_incoming/rent3d.tar.gz\n"
        "and re-run; it will be verified and extracted automatically.\n\n"
        "Note: Rent3D++ (richer annotations, same apartments) may be easier to get -\n"
        "see the 'rent3dpp' entry. See docs/PRIOR-ART.md for why we want this."),
    homepage="https://www.cs.toronto.edu/~fidler/projects/rent3D.html",
    citation=("Liu, Schwing, Kundu, Urtasun, Fidler. Rent3D: Floor-Plan Priors for "
              "Monocular Layout Estimation. CVPR 2015."),
    approx_size="~1 GB",
    expect=("rent3d",),
))

register(DatasetSpec(
    name="rent3dpp",
    description="Rent3D++ (Plan2Scene): same 215 apartments, richer annotations plus the "
                "coverage-level evaluation protocol we adopt for metric M8.",
    urls=(),
    licence="Non-commercial use stated on request; Plan2Scene code is MIT",
    verdict="licence_unclear",
    acquisition="manual",
    manual_instructions=(
        "Rent3D++ is distributed by request, not direct download.\n"
        "  1. Request access: https://forms.gle/mKAmnrzAm3LCK9ua6\n"
        "  2. When granted, place the archive at:\n"
        "       $VISITIT_DATA_HOME/_incoming/rent3dpp.zip\n"
        "  3. Re-run this command; it will verify and extract it.\n"
        "See docs/PRIOR-ART.md section 6."),
    homepage="https://github.com/3dlg-hcvc/plan2scene",
    expect=("rent3dpp",),
))

# ------------------------------------------- commercially-usable corpora
register(DatasetSpec(
    name="resplan",
    description="17,000 vector residential floor plans with room-connectivity graphs "
                "and metric scale. Primary vectoriser pretraining corpus.",
    urls=(
        "https://codeload.github.com/m-agour/ResPlan/zip/refs/heads/main",
        "https://github.com/m-agour/ResPlan/archive/refs/heads/main.zip",
    ),
    licence="CC BY 4.0 (attribution required)",
    verdict="prod_ok",
    manual_instructions=(
        "GitHub is unreachable from repo-scoped CI sandboxes (403). On a normal\n"
        "machine the direct URLs work. Otherwise download the repo zip or the\n"
        "Kaggle mirror and place it at $VISITIT_DATA_HOME/_incoming/resplan.zip"),
    homepage="https://github.com/m-agour/ResPlan",
    approx_size="~200 MB",
    expect=("resplan",),
))

register(DatasetSpec(
    name="swiss_dwellings",
    description="42,207 apartments / 242,257 rooms across 3,093 European buildings, "
                "with geometry. Second vectoriser corpus.",
    urls=(
        "https://zenodo.org/records/7070952/files/swiss-dwellings-v3.0.0.zip?download=1",
        "https://zenodo.org/record/7070952/files/swiss-dwellings-v3.0.0.zip",
    ),
    licence="CC BY 4.0 (attribute Archilyse AG)",
    verdict="prod_ok",
    manual_instructions=(
        "Zenodo returns 403 to this sandbox's egress proxy; it downloads fine from\n"
        "a normal network. Otherwise fetch it manually and place the zip at\n"
        "$VISITIT_DATA_HOME/_incoming/swiss_dwellings.zip"),
    homepage="https://zenodo.org/records/7070952",
    approx_size="~2.3 GB",
    expect=("swiss_dwellings",),
))
