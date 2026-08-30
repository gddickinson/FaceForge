"""Named, ordered study sets ("curricula") derived from the project's own data.

Two sources, no hand-written membership lists:

1. **Config groups.**  Every asset config under ``assets/config`` is a list of
   ``{"name": ..., "stl": "FMA52734", ...}`` entries, and the file it lives in
   *is* an anatomical grouping the project already maintains:
   ``muscles/expression_muscles.json`` is the muscles of facial expression,
   ``skull_bones.json`` is the skull, ``skeleton/rib_cage.json`` is the rib
   cage.  Each file becomes one curriculum.  A curriculum therefore cannot
   drift out of sync with what the application can actually load -- if a mesh
   is added to a config, it joins the curriculum.
2. **FMA body systems.**  ``assets/config/fma_labels.json`` carries a
   ``system`` per structure, derived by walking the FMA parent chain, so
   ``system:muscular`` and ``system:skeletal`` are curricula spanning config
   groups.

Difficulty tiers
----------------
Tier comes from the **token count of the FMA preferred term**, which is an
objective property of the ontology rather than an opinion about what is hard:

=============  =============  ====================================
Tier           Label tokens   Example
=============  =============  ====================================
foundation     1-2            ``Cerebellum``, ``Frontal bone``
intermediate   3              ``Ninth thoracic vertebra``
advanced       4+             ``Deep part of right masseter``
unclassified   (no FMA term)  config entry absent from the crosswalk
=============  =============  ====================================

The justification is that FMA preferred terms are compositional: a one-word
term is a whole organ or a named muscle, and each additional qualifier names a
part, a side, or a member of a numbered series -- i.e. a finer distinction the
learner has to make.  It is a proxy, and it is stated here as one; it is not a
claim that every 4-token structure is harder than every 3-token structure.

Ordering *within* a tier is by (token count, then preferred term, then id) --
fully deterministic, so a curriculum renders the same on every machine and a
test can assert on position.

What is deliberately absent
---------------------------
There is no "cranial nerves" curriculum.  The crosswalk contains exactly two
structures whose FMA system is ``nervous`` and whose preferred term contains
"nerve" (left and right optic nerve): the BodyParts3D distribution used here
does not ship cranial nerve meshes, so such a curriculum would be a two-item
list of one nerve.  Rather than hardcode names with no meshes behind them,
:func:`build_curricula` builds only what the data supports and
:func:`missing_topics` reports what it could not build and why.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from faceforge.constants import CONFIG_DIR

logger = logging.getLogger(__name__)

#: Tier names in study order.  ``unclassified`` is last: an item with no FMA
#: term has no defensible position, so it is asked after everything that does.
TIER_ORDER = ("foundation", "intermediate", "advanced", "unclassified")

#: Token-count boundaries. Read as: <= 2 tokens -> foundation, == 3 ->
#: intermediate, else advanced.
_FOUNDATION_MAX_TOKENS = 2
_INTERMEDIATE_MAX_TOKENS = 3

#: Cosmetic only -- affects the title shown, never the membership, which is
#: always the config file's contents.  Anything not listed gets a title
#: mechanically derived from the file stem.
_TITLE_OVERRIDES = {
    "expression_muscles": "Muscles of facial expression",
    "skull_bones": "Skull bones",
    "cns_additional": "Central nervous system (additional)",
    "cardiac_additional": "Cardiac structures (additional)",
    "vascular": "Blood vessels",
    "oral": "Oral cavity",
    "pelvic_floor": "Pelvic floor",
    "face_features": "Facial soft-tissue features",
}

#: Config files that are not structure lists (parameters, transforms, colour
#: tables).  Detected structurally rather than by name -- see _iter_entries --
#: but listed here so the intent is documented.
_NON_STRUCTURE_HINT = (
    "au_definitions", "body_joint_limits", "body_poses", "coordinate_transform",
    "expressions", "eye_colors", "fma_labels", "gender_dimorphism",
    "joint_limits", "muscle_dof_map", "pathology_presets", "skinning_overrides",
    "face_regions",
)


@dataclass(frozen=True)
class CurriculumItem:
    """One structure in a study set."""

    item_id: str              # BodyParts3D mesh id, e.g. "FMA52734"
    display_name: str         # what the app calls it
    preferred_label: str = ""  # FMA preferred term ("" if not in crosswalk)
    system: str = ""          # FMA body system ("" if unknown)
    category: str = ""        # owning config group per the crosswalk
    tier: str = "unclassified"

    @property
    def label(self) -> str:
        """Best available human term."""
        return self.preferred_label or self.display_name


@dataclass(frozen=True)
class Curriculum:
    """A named, ordered set of structures with difficulty tiers."""

    key: str
    title: str
    source: str               # "config_group" | "fma_system"
    items: tuple[CurriculumItem, ...] = ()

    def __len__(self) -> int:
        return len(self.items)

    @property
    def tiers(self) -> tuple[str, ...]:
        """Tiers actually present, in study order."""
        present = {i.tier for i in self.items}
        return tuple(t for t in TIER_ORDER if t in present)

    def tier_items(self, tier: str) -> tuple[CurriculumItem, ...]:
        return tuple(i for i in self.items if i.tier == tier)

    def item_ids(self, tier: str = "") -> list[str]:
        """Ordered ids, optionally restricted to one tier."""
        rows = self.items if not tier else self.tier_items(tier)
        return [i.item_id for i in rows]

    def counts(self) -> dict[str, int]:
        return {t: len(self.tier_items(t)) for t in self.tiers}


def tier_for_label(preferred_label: str) -> str:
    """Difficulty tier from the FMA preferred term (see module docstring)."""
    if not preferred_label.strip():
        return "unclassified"
    tokens = len(preferred_label.split())
    if tokens <= _FOUNDATION_MAX_TOKENS:
        return "foundation"
    if tokens <= _INTERMEDIATE_MAX_TOKENS:
        return "intermediate"
    return "advanced"


def _title_for(stem: str) -> str:
    if stem in _TITLE_OVERRIDES:
        return _TITLE_OVERRIDES[stem]
    words = stem.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def _iter_entries(data: object) -> Iterable[dict]:
    """Yield ``{"name", "stl"}`` dicts from a loaded config, at any nesting.

    Structural, not name-based: a config is a structure list if it contains
    entries with both a name and an ``stl`` mesh id.  Parameter files
    (joint limits, AU definitions) contain no such entries and yield nothing.
    """
    if isinstance(data, dict):
        if isinstance(data.get("stl"), str) and isinstance(data.get("name"), str):
            yield data
            return
        for value in data.values():
            yield from _iter_entries(value)
    elif isinstance(data, list):
        for value in data:
            yield from _iter_entries(value)


def _config_files(config_dir: Path) -> list[Path]:
    """Every config JSON, top level plus the muscles/ and skeleton/ subdirs."""
    files = sorted(config_dir.glob("*.json"))
    for sub in ("muscles", "skeleton"):
        files.extend(sorted((config_dir / sub).glob("*.json")))
    return files


def _make_item(mesh_id: str, display_name: str, fma: dict) -> CurriculumItem:
    meta = fma.get(mesh_id) or {}
    label = str(meta.get("preferred_label") or "")
    return CurriculumItem(
        item_id=mesh_id,
        display_name=display_name,
        preferred_label=label,
        system=str(meta.get("system") or ""),
        category=str(meta.get("category") or ""),
        tier=tier_for_label(label),
    )


def _ordered(items: Iterable[CurriculumItem]) -> tuple[CurriculumItem, ...]:
    """Tier order, then label token count, then label, then id."""
    return tuple(sorted(
        items,
        key=lambda i: (
            TIER_ORDER.index(i.tier) if i.tier in TIER_ORDER else len(TIER_ORDER),
            len(i.label.split()),
            i.label.lower(),
            i.item_id,
        ),
    ))


def build_curricula(
    fma: Optional[dict] = None,
    config_dir: Optional[Path] = None,
) -> dict[str, Curriculum]:
    """Build every curriculum the data supports.

    Parameters
    ----------
    fma:
        The crosswalk (mesh id -> metadata).  Loaded from
        ``assets/config/fma_labels.json`` when omitted.
    config_dir:
        Root of the asset configs.  Defaults to the project's ``CONFIG_DIR``;
        tests pass a temporary directory.

    Returns
    -------
    dict
        ``key -> Curriculum``.  Config-group keys are the file stems
        (``"expression_muscles"``); system keys are prefixed
        (``"system:muscular"``).  Empty curricula are dropped.
    """
    if fma is None:
        from faceforge.loaders.stl_batch_loader import load_fma_labels
        fma = load_fma_labels()
    root = Path(config_dir) if config_dir is not None else CONFIG_DIR

    out: dict[str, Curriculum] = {}
    seen_ids: dict[str, CurriculumItem] = {}

    for path in _config_files(root):
        stem = path.stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Curriculum source %s could not be read; skipped", path)
            continue

        items: dict[str, CurriculumItem] = {}
        for entry in _iter_entries(data):
            mesh_id = entry["stl"]
            item = _make_item(mesh_id, entry["name"], fma)
            items[mesh_id] = item
            seen_ids.setdefault(mesh_id, item)
        if not items:
            if stem not in _NON_STRUCTURE_HINT:
                logger.debug("Config %s has no structure entries", path)
            continue
        out[stem] = Curriculum(
            key=stem,
            title=_title_for(stem),
            source="config_group",
            items=_ordered(items.values()),
        )

    # FMA-system curricula, over every structure any config references.
    by_system: dict[str, list[CurriculumItem]] = {}
    for item in seen_ids.values():
        if item.system:
            by_system.setdefault(item.system, []).append(item)
    for system, items in sorted(by_system.items()):
        key = f"system:{system}"
        out[key] = Curriculum(
            key=key,
            title=f"{system.capitalize()} system",
            source="fma_system",
            items=_ordered(items),
        )

    return out


def missing_topics(curricula: dict[str, Curriculum]) -> dict[str, str]:
    """Topics a learner might expect that the dataset cannot support.

    Returned as ``topic -> reason``, computed from ``curricula`` rather than
    asserted, so it stays true if the dataset changes.
    """
    out: dict[str, str] = {}
    nervous = curricula.get("system:nervous")
    nerves = [
        i for i in (nervous.items if nervous else ())
        if "nerve" in i.label.lower()
    ]
    if len(nerves) < 12:
        out["cranial nerves"] = (
            f"only {len(nerves)} nerve structure(s) in the dataset "
            f"({', '.join(sorted(i.label for i in nerves)) or 'none'}); "
            "the 12 cranial nerves are not shipped as meshes"
        )
    return out


_CACHE: Optional[dict[str, Curriculum]] = None


def get_curricula(refresh: bool = False) -> dict[str, Curriculum]:
    """Cached :func:`build_curricula` over the project's real configs."""
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = build_curricula()
    return _CACHE
