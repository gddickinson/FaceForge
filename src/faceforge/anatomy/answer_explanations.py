"""Why a wrong quiz answer was wrong, in real anatomical terms.

"Incorrect. Answer: Temporalis R" tells a learner nothing they did not
already know.  This module resolves *both* the structure they chose and the
structure that was correct against the FMA crosswalk
(``assets/config/fma_labels.json``) and states the FMA preferred term of each
plus every attribute on which they differ: body system, owning config group,
anatomical region and laterality.

Every field comes from the crosswalk or from the region inference already used
by :mod:`faceforge.anatomy.structure_search`.  There is no canned prose keyed
by structure name, and no text is emitted for an attribute the data does not
carry -- an unknown system is reported as unknown rather than guessed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

_LEFT = re.compile(r"\b(left|l)\b|\bL$", re.IGNORECASE)
_RIGHT = re.compile(r"\b(right|r)\b|\bR$", re.IGNORECASE)


@dataclass(frozen=True)
class StructureFacts:
    """Everything known about one structure, from real data only."""

    item_id: str = ""
    display_name: str = ""
    preferred_label: str = ""
    system: str = ""
    category: str = ""
    region: str = ""
    side: str = ""            # "left" | "right" | "" (midline/unspecified)

    @property
    def known(self) -> bool:
        return bool(self.item_id or self.display_name)

    @property
    def label(self) -> str:
        return self.preferred_label or self.display_name

    def describe(self) -> str:
        """One clause: the preferred term plus the attributes that are known."""
        if not self.known:
            return "an unrecognised structure"
        bits = []
        if self.system:
            bits.append(f"{self.system} system")
        if self.category:
            bits.append(f"{self.category} group")
        if self.region:
            bits.append(f"{self.region} region")
        if self.side:
            bits.append(self.side)
        detail = ", ".join(bits) if bits else "no system recorded in the crosswalk"
        return f"{self.label} ({detail})"


@dataclass(frozen=True)
class Explanation:
    """A wrong-answer explanation.

    ``differences`` is the list of attributes on which the two structures
    actually differ, each as ``(attribute, chosen_value, correct_value)`` with
    ``""`` for unknown.  ``text`` is the rendering shown in the UI; callers
    that want to lay it out differently should use the structured fields.
    """

    chosen: StructureFacts
    correct: StructureFacts
    differences: tuple[tuple[str, str, str], ...] = ()
    text: str = ""

    @property
    def same_system(self) -> bool:
        return bool(self.chosen.system) and self.chosen.system == self.correct.system


class ExplanationBuilder:
    """Resolves answers to structures and explains the difference.

    Parameters
    ----------
    fma:
        The crosswalk (mesh id -> metadata).  Loaded from the project's
        ``fma_labels.json`` when omitted.
    """

    def __init__(self, fma: Optional[dict] = None):
        if fma is None:
            from faceforge.loaders.stl_batch_loader import load_fma_labels
            fma = load_fma_labels()
        self._fma = fma or {}
        # name -> mesh id, over both the display name and the FMA term, so a
        # learner who types the standard term is resolved as precisely as one
        # who types what the UI shows.
        self._by_name: dict[str, str] = {}
        for mesh_id, meta in self._fma.items():
            for key in (meta.get("display_name"), meta.get("preferred_label")):
                if key:
                    self._by_name.setdefault(str(key).strip().lower(), mesh_id)

    # -- resolution --------------------------------------------------------

    def facts(self, answer: str) -> StructureFacts:
        """Resolve a mesh id, display name or FMA term to known facts.

        Falls back to a fuzzy name match (ratio >= 0.85) so a near-miss
        spelling still yields a real structure rather than "unrecognised";
        below that threshold the answer is reported as unrecognised instead of
        being forced onto the closest entry.
        """
        text = (answer or "").strip()
        if not text:
            return StructureFacts()

        mesh_id = text if text in self._fma else self._by_name.get(text.lower())
        if mesh_id is None:
            mesh_id = self._fuzzy_id(text.lower())
        if mesh_id is None:
            # Unknown to the crosswalk, but the learner still typed something;
            # keep it so the explanation can quote it back.
            return StructureFacts(display_name=text, side=_side_of(text))

        meta = self._fma.get(mesh_id, {})
        display = str(meta.get("display_name") or "")
        label = str(meta.get("preferred_label") or "")
        return StructureFacts(
            item_id=mesh_id,
            display_name=display,
            preferred_label=label,
            system=str(meta.get("system") or ""),
            category=str(meta.get("category") or ""),
            region=_region_of(f"{display} {label}"),
            side=_side_of(label or display),
        )

    def _fuzzy_id(self, needle: str) -> Optional[str]:
        best, best_ratio = None, 0.0
        for name, mesh_id in self._by_name.items():
            ratio = SequenceMatcher(None, needle, name).ratio()
            if ratio > best_ratio:
                best, best_ratio = mesh_id, ratio
        return best if best_ratio >= 0.85 else None

    # -- explanation -------------------------------------------------------

    def explain(self, chosen_answer: str, correct_answer: str) -> Explanation:
        """Explain why ``chosen_answer`` is not ``correct_answer``."""
        chosen = self.facts(chosen_answer)
        correct = self.facts(correct_answer)
        diffs = _differences(chosen, correct)
        return Explanation(
            chosen=chosen,
            correct=correct,
            differences=diffs,
            text=_render(chosen, correct, diffs),
        )


def _differences(a: StructureFacts, b: StructureFacts
                 ) -> tuple[tuple[str, str, str], ...]:
    out = []
    for attr in ("system", "category", "region", "side"):
        av, bv = getattr(a, attr), getattr(b, attr)
        if av != bv:
            out.append((attr, av, bv))
    return tuple(out)


_ATTR_WORDS = {
    "system": "body system",
    "category": "config group",
    "region": "anatomical region",
    "side": "side",
}


def _render(chosen: StructureFacts, correct: StructureFacts,
            diffs: tuple[tuple[str, str, str], ...]) -> str:
    if not correct.known:
        return "No answer was recorded for this question."

    if not chosen.known:
        return (
            f"No answer given. The structure is {correct.describe()}."
        )

    if chosen.item_id and chosen.item_id == correct.item_id:
        return f"That is the same structure: {correct.describe()}."

    if not chosen.item_id:
        return (
            f"\u201c{chosen.display_name}\u201d does not match any structure in "
            f"this model. The answer is {correct.describe()}."
        )

    head = (
        f"You chose {chosen.describe()}; the answer is {correct.describe()}."
    )
    if not diffs:
        # Same system, group, region and side: the distinction is the FMA term
        # itself, so give both ids for a lookup rather than inventing prose.
        return (
            f"{head} They share body system, group, region and side \u2014 the "
            f"distinction is the term itself: {chosen.item_id} "
            f"{chosen.label!r} vs {correct.item_id} {correct.label!r}."
        )
    parts = [
        f"{_ATTR_WORDS[attr]} ({av or 'unknown'} vs {bv or 'unknown'})"
        for attr, av, bv in diffs
    ]
    return f"{head} They differ in " + "; ".join(parts) + "."


def _region_of(text: str) -> str:
    """Region via the search index's inference, so one rule serves both."""
    try:
        from faceforge.anatomy.structure_search import AnatomySearchIndex
    except ImportError:            # pragma: no cover - circular-import guard
        return ""
    return AnatomySearchIndex._infer_region_from_name(text)


def _side_of(text: str) -> str:
    if _LEFT.search(text or ""):
        return "left"
    if _RIGHT.search(text or ""):
        return "right"
    return ""
