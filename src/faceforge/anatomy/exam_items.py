"""Exam item schema, provenance, and the gate that refuses ungrounded items.

The design constraint is a safety property, not a style choice: **a learner
cannot tell a plausible wrong anatomical fact from a right one**, so a fact
that is not traceable to real data must never reach them.  The schema enforces
that structurally rather than by convention:

* every :class:`ExamItem` carries at least one :class:`Provenance` naming the
  file, FMA id or relation the fact came from;
* every item carries ``verified``, which is True only when *every* provenance
  entry is machine-derived (see :data:`DERIVED_SOURCES`);
* :func:`present` -- the only sanctioned way to show an item -- raises
  :class:`ItemRefused` in exam mode for an item that is unverified, has no
  provenance, or (for authored content) has no citation.

So a question type can be scaffolded without content: the renderer refuses the
empty case, and the refusal is a tested behaviour rather than a promise.

Levels
------
Named for what they test, each tied to the data that powers it:

======  ==============================  =========================================
Level   Tests                           Powered by
======  ==============================  =========================================
L1      gross identification            mesh + ``fma_labels.json`` preferred term
L2      systems / regions / laterality  ``fma_labels.json`` system + category
L3      hierarchical relations          ``fma_taxonomy.json`` is-a / part-of
L4      radiological cross-section      ``scanner/engine.py`` render + L1 labels
L5      clinical reasoning (format)      authored content + citation, none shipped
======  ==============================  =========================================

Formats
-------
``sba``      single best answer, configurable option count (5 is the USMLE norm)
``spot``     tag identification against the 3D render
``emq``      extended matching: one option list, several stems
``station``  timed OSPE/steeplechase station: fixed seconds, no going back
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

LEVELS = ("L1", "L2", "L3", "L4", "L5")

LEVEL_TITLES = {
    "L1": "Gross identification",
    "L2": "Systems, regions and laterality",
    "L3": "Hierarchical relations",
    "L4": "Radiological cross-section identification",
    "L5": "Clinical reasoning",
}

LEVEL_SOURCES = {
    "L1": "mesh geometry + assets/config/fma_labels.json",
    "L2": "assets/config/fma_labels.json (system, category)",
    "L3": "assets/config/fma_taxonomy.json (is-a, part-of, composite-of)",
    "L4": "faceforge.scanner.engine render + assets/config/fma_labels.json",
    "L5": "externally authored content with a citation (none shipped)",
}

FORMATS = ("sba", "spot", "emq", "station")

#: Provenance kinds that are machine-derived from data in this repository or
#: the BodyParts3D distribution.  An item whose provenance is entirely from
#: this set is ``verified``.
DERIVED_SOURCES = frozenset({
    "fma_label",         # fma_labels.json row for a mesh id
    "fma_is_a",          # FMA.csv subClassOf edge, via fma_taxonomy.json
    "fma_part_of",       # conventional_part_of.txt, via fma_taxonomy.json
    "fma_composite_of",  # composite_parts.txt, via fma_taxonomy.json
    "config_file",       # an assets/config/**.json entry
    "scanner_render",    # a scan produced by faceforge.scanner.engine
})

#: Provenance kinds that are human-authored and therefore require a citation.
AUTHORED_SOURCES = frozenset({"citation"})


class ItemRefused(Exception):
    """Raised when an item is not fit to present.  Carries the reasons."""

    def __init__(self, item_uid: str, reasons: Sequence[str]):
        self.item_uid = item_uid
        self.reasons = list(reasons)
        super().__init__(f"item {item_uid!r} refused: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class Provenance:
    """Where one fact in an item came from.

    ``kind`` is a member of :data:`DERIVED_SOURCES` or
    :data:`AUTHORED_SOURCES`; ``reference`` identifies the row, id or file;
    ``detail`` is free text for a human reading the audit trail.
    """

    kind: str
    reference: str
    detail: str = ""

    @property
    def derived(self) -> bool:
        return self.kind in DERIVED_SOURCES

    def __str__(self) -> str:
        base = f"{self.kind}:{self.reference}"
        return f"{base} ({self.detail})" if self.detail else base


@dataclass(frozen=True)
class Option:
    """One answer option.

    ``role`` records *why* this option is in the list -- ``"answer"``, or the
    neighbourhood a distractor was drawn from (``"is_a_sibling"``,
    ``"is_a_cousin"``, ``"same_system"``, ...).  Recording it is what makes
    "are the distractors anatomically adjacent?" a testable question rather
    than a claim.
    """

    text: str
    item_id: str = ""
    role: str = "distractor"
    provenance: tuple[Provenance, ...] = ()


@dataclass(frozen=True)
class ExamItem:
    """A single exam item.

    ``uid`` is a deterministic hash of the item's content, so the same
    generator inputs always produce the same id and a learner's per-item
    statistics survive regeneration.
    """

    level: str
    fmt: str
    stem: str
    options: tuple[Option, ...]
    answer_index: int
    focus_id: str = ""
    provenance: tuple[Provenance, ...] = ()
    citation: str = ""
    tags: tuple[str, ...] = ()
    seconds: float = 0.0
    #: Set for spot/L4 items: where on the image the tagged structure is,
    #: in fractional image coordinates (x, y), origin top-left.
    tag_xy: Optional[tuple[float, float]] = None
    uid: str = ""

    def __post_init__(self):
        if not self.uid:
            object.__setattr__(self, "uid", self.compute_uid())

    def compute_uid(self) -> str:
        payload = "|".join([
            self.level, self.fmt, self.stem, self.focus_id,
            *(f"{o.item_id}~{o.text}" for o in self.options),
            str(self.answer_index),
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    # -- derived properties -----------------------------------------------

    @property
    def answer(self) -> Optional[Option]:
        if 0 <= self.answer_index < len(self.options):
            return self.options[self.answer_index]
        return None

    @property
    def verified(self) -> bool:
        """True when every provenance entry is machine-derived.

        An item with no provenance at all is *not* verified -- absence of a
        claim about sourcing is not evidence of sourcing.
        """
        all_prov = list(self.provenance) + [
            p for o in self.options for p in o.provenance]
        return bool(all_prov) and all(p.derived for p in all_prov)

    @property
    def distractor_roles(self) -> tuple[str, ...]:
        return tuple(o.role for i, o in enumerate(self.options)
                     if i != self.answer_index)

    # -- validation --------------------------------------------------------

    def problems(self, exam_mode: bool = True, min_options: int = 2) -> list[str]:
        """Everything wrong with this item.  Empty list means presentable."""
        out: list[str] = []
        if self.level not in LEVELS:
            out.append(f"unknown level {self.level!r}")
        if self.fmt not in FORMATS:
            out.append(f"unknown format {self.fmt!r}")
        if not self.stem.strip():
            out.append("empty stem")
        if len(self.options) < min_options:
            out.append(f"{len(self.options)} option(s), need {min_options}")
        if not 0 <= self.answer_index < len(self.options):
            out.append("answer_index out of range")
        texts = [o.text.strip().lower() for o in self.options]
        if len(set(texts)) != len(texts):
            out.append("duplicate option text")
        if any(not t for t in texts):
            out.append("blank option text")
        if not self.provenance and not any(o.provenance for o in self.options):
            out.append("no provenance")
        if exam_mode:
            if not self.verified:
                authored = [p for p in self.provenance if not p.derived]
                if authored and not self.citation.strip():
                    out.append("authored content without a citation")
                elif not self.citation.strip():
                    out.append("unverified and uncited")
        return out

    def with_options(self, options: Iterable[Option], answer_index: int
                     ) -> "ExamItem":
        """Copy carrying different options (used when shuffling)."""
        return replace(replace(self, options=tuple(options),
                               answer_index=answer_index), uid="")

    # -- audit -------------------------------------------------------------

    def provenance_report(self) -> str:
        """Human-readable audit trail: every fact and where it came from."""
        lines = [f"{self.uid} [{self.level}/{self.fmt}] verified={self.verified}"]
        lines += [f"  item: {p}" for p in self.provenance]
        for i, opt in enumerate(self.options):
            marker = "*" if i == self.answer_index else " "
            lines.append(f"  {marker} {opt.text} [{opt.role}]")
            lines += [f"      {p}" for p in opt.provenance]
        if self.citation:
            lines.append(f"  citation: {self.citation}")
        return "\n".join(lines)


def present(item: ExamItem, exam_mode: bool = True,
            min_options: int = 2) -> ExamItem:
    """Return ``item`` if it is fit to show, else raise :class:`ItemRefused`.

    This is the gate.  Callers must route every item through it rather than
    reading ``item.stem`` directly, which is why it returns the item: the
    calling code reads no worse for being checked.
    """
    problems = item.problems(exam_mode=exam_mode, min_options=min_options)
    if problems:
        raise ItemRefused(item.uid, problems)
    return item


def presentable(items: Iterable[ExamItem], exam_mode: bool = True
                ) -> tuple[list[ExamItem], list[tuple[str, list[str]]]]:
    """Split ``items`` into (presentable, [(uid, reasons), ...]).

    Used by session assembly so a bad generator drops items instead of
    aborting a whole exam, and so the drop is reportable.
    """
    good: list[ExamItem] = []
    bad: list[tuple[str, list[str]]] = []
    for item in items:
        reasons = item.problems(exam_mode=exam_mode)
        (good if not reasons else bad).append(
            item if not reasons else (item.uid, reasons))
    return good, bad
