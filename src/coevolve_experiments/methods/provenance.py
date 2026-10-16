"""Pinned clean-room provenance for adapted comparison methods.

No upstream source is imported or vendored.  The adapters encode only the
paper-level collaboration mechanisms over CoEvolve's common physical contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptationProvenance:
    method: str
    repository_url: str
    repository_commit: str
    upstream_paper_revision: str
    local_paper_revision: str
    local_paper_sha256: str
    adaptation_basis: str = "paper-level common-core clean-room adaptation"


REMAC_REPOSITORY_COMMIT = "8344151d5e6f5a3e1d6ea36530a274f97542bbbd"
COHERENT_REPOSITORY_COMMIT = "17554a52792e302921bbb5e5fd3b4049e25b50d4"

REMAC_PROVENANCE = AdaptationProvenance(
    method="REMAC",
    repository_url="https://github.com/reamac-repo/remac-repo",
    repository_commit=REMAC_REPOSITORY_COMMIT,
    upstream_paper_revision="arXiv:2503.22122v2",
    local_paper_revision="arXiv:2503.22122v1 (8 pages)",
    local_paper_sha256=(
        "439830f5d8b6d0ed270d55f683e1e56a1445856be19042448f2086ec8b06d8e6"
    ),
)

COHERENT_PROVENANCE = AdaptationProvenance(
    method="COHERENT",
    repository_url="https://github.com/MrKeee/COHERENT",
    repository_commit=COHERENT_REPOSITORY_COMMIT,
    upstream_paper_revision="arXiv:2409.15146v3 / ICRA",
    local_paper_revision="arXiv:2409.15146v3 (7 pages / ICRA)",
    local_paper_sha256=(
        "3780db471bf44191bf8ad468bcea2fa220d8f3949fe69a8849a22c21ec654731"
    ),
)

