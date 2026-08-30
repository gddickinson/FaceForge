"""Distractor selection: anatomically adjacent wrong answers, deterministically.

Distractor quality is what separates an exam item from a giveaway.  An
identification item that offers ``femur`` against a skull bone is free marks;
one that offers the four other bones of the neurocranium discriminates between
a student who knows the region and one who does not.

The neighbourhood ladder, tried in order, each rung documented because each is
a different claim about "adjacent":

1. **is-a siblings** -- structures sharing the focus structure's immediate FMA
   superclass.  ``Frontal bone`` yields parietal, occipital, temporal.  This is
   the strongest rung and often the only one an examiner would accept.
2. **is-a cousins** -- structures sharing the grandparent class.  Needed
   because sibling groups in this dataset have a median size of 2, so rung 1
   alone cannot fill a five-option item for most structures.
3. **shared containing whole** -- structures that the part-of and composite
   tables place inside the same narrowest whole as the focus.  ``Right
   temporalis`` has exactly one is-a sibling (the left temporalis) and no
   cousins, but it shares ``musculature of head`` with every other muscle of
   the head, which is the neighbourhood an examiner would actually use.
4. **same system and category** -- from ``fma_labels.json``.
5. **same system** -- broader still.
6. **same category** -- the config group, when the system is unknown.

Determinism: selection is seeded (``random.Random(seed)``) and every candidate
list is sorted before sampling, so an item generated from the same focus and
seed has the same distractors on every machine and in every process.  That is
what makes distractor provenance testable.

Nothing here invents a structure: every candidate is a real id from the FMA
taxonomy or the crosswalk, and each returned distractor says which rung it
came from.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: Ladder rungs in the order they are tried, with the role recorded on the
#: option and the provenance kind that justifies it.
LADDER = (
    ("is_a_sibling", "fma_is_a"),
    ("is_a_cousin", "fma_is_a"),
    ("shares_whole", "fma_part_of"),
    ("same_system_and_category", "fma_label"),
    ("same_system", "fma_label"),
    ("same_category", "fma_label"),
)


@dataclass(frozen=True)
class Distractor:
    """A wrong option: which structure, and which rung it was drawn from."""

    item_id: str
    label: str
    role: str
    provenance_kind: str
    provenance_reference: str


class DistractorPool:
    """Chooses distractors for a focus structure.

    Parameters
    ----------
    fma:
        The crosswalk (mesh id -> metadata).  Loaded from the project asset
        when omitted.
    taxonomy:
        A :class:`~faceforge.anatomy.fma_taxonomy.Taxonomy`.  Loaded from the
        project asset when omitted.
    restrict_to:
        Optional set of ids the distractors must come from -- pass the
        curriculum's ids to keep an exam within the material studied.  Ids
        outside the crosswalk are always excluded, since their label would be
        unknown.
    """

    def __init__(self, fma: Optional[dict] = None, taxonomy=None,
                 restrict_to: Optional[Iterable[str]] = None):
        if fma is None:
            from faceforge.loaders.stl_batch_loader import load_fma_labels
            fma = load_fma_labels()
        if taxonomy is None:
            from faceforge.anatomy.fma_taxonomy import get_taxonomy
            taxonomy = get_taxonomy()
        self._fma = fma or {}
        self._tax = taxonomy
        self._allowed = set(restrict_to) if restrict_to is not None else None

        # Index the crosswalk once per pool.
        self._wholes: dict[str, tuple[str, ...]] = {}
        self._by_whole: dict[str, list[str]] = {}
        self._by_system: dict[str, list[str]] = {}
        self._by_category: dict[str, list[str]] = {}
        self._by_system_category: dict[tuple[str, str], list[str]] = {}
        for mesh_id, meta in sorted(self._fma.items()):
            system = str(meta.get("system") or "")
            category = str(meta.get("category") or "")
            if system:
                self._by_system.setdefault(system, []).append(mesh_id)
            if category:
                self._by_category.setdefault(category, []).append(mesh_id)
            if system and category:
                self._by_system_category.setdefault(
                    (system, category), []).append(mesh_id)
            containing = tuple(dict.fromkeys(
                list(self._tax.part_of(mesh_id)) + list(self._tax.wholes(mesh_id))))
            if containing:
                self._wholes[mesh_id] = containing
                for whole in containing:
                    self._by_whole.setdefault(whole, []).append(mesh_id)

    # -- labels ------------------------------------------------------------

    def label(self, mesh_id: str) -> str:
        """Preferred FMA term for an id, falling back to the taxonomy."""
        meta = self._fma.get(mesh_id) or {}
        return str(meta.get("preferred_label") or "") or self._tax.label(mesh_id)

    # -- candidate rungs ---------------------------------------------------

    def _candidates(self, role: str, focus_id: str) -> list[str]:
        meta = self._fma.get(focus_id) or {}
        system = str(meta.get("system") or "")
        category = str(meta.get("category") or "")
        if role == "is_a_sibling":
            return self._tax.siblings(focus_id)
        if role == "is_a_cousin":
            return self._tax.cousins(focus_id)
        if role == "shares_whole":
            return self._whole_candidates(focus_id)
        if role == "same_system_and_category":
            return list(self._by_system_category.get((system, category), ()))
        if role == "same_system":
            return list(self._by_system.get(system, ()))
        if role == "same_category":
            return list(self._by_category.get(category, ()))
        return []

    def _acceptable(self, mesh_id: str, focus_id: str,
                    used_labels: set[str]) -> bool:
        if mesh_id == focus_id:
            return False
        if self._allowed is not None and mesh_id not in self._allowed:
            return False
        label = self.label(mesh_id)
        if not label:
            # Without a label there is nothing to show, and inventing one is
            # exactly what this module must not do.
            return False
        return label.strip().lower() not in used_labels

    # -- selection ---------------------------------------------------------

    def choose(self, focus_id: str, count: int, seed: int = 0,
               exclude: Iterable[str] = ()) -> list[Distractor]:
        """Up to ``count`` distractors for ``focus_id``, best rung first.

        Sampling is seeded and the candidate list is sorted first, so the
        result is reproducible.  Fewer than ``count`` are returned when the
        data cannot supply them -- the caller decides whether an item with
        three options is worth asking, rather than this function padding the
        list with something unrelated.
        """
        rng = random.Random(f"{focus_id}|{seed}|{count}")
        focus_label = self.label(focus_id).strip().lower()
        used_labels = {focus_label} | {
            self.label(e).strip().lower() for e in exclude if self.label(e)
        }
        used_ids = {focus_id} | set(exclude)

        out: list[Distractor] = []
        for role, prov_kind in LADDER:
            if len(out) >= count:
                break
            pool = sorted({
                c for c in self._candidates(role, focus_id)
                if c not in used_ids and self._acceptable(c, focus_id, used_labels)
            })
            if not pool:
                continue
            take = min(count - len(out), len(pool))
            for mesh_id in rng.sample(pool, take):
                label = self.label(mesh_id)
                out.append(Distractor(
                    item_id=mesh_id,
                    label=label,
                    role=role,
                    provenance_kind=prov_kind,
                    provenance_reference=self._reference(role, focus_id, mesh_id),
                ))
                used_ids.add(mesh_id)
                used_labels.add(label.strip().lower())
        return out

    def _narrowest_whole(self, focus_id: str) -> str:
        """The narrowest structure the part-of tables place ``focus_id`` in."""
        return self._tax.most_specific(self._wholes.get(focus_id, ()))

    #: Widening stops once this many candidates are available, and a whole
    #: with more members than this is skipped entirely -- "part of the human
    #: body" is true of everything and so discriminates nothing.
    _WHOLE_TARGET = 12
    _WHOLE_MAX_MEMBERS = 150

    def _ordered_wholes(self, focus_id: str) -> list[str]:
        """The focus structure's containing wholes, narrowest first."""
        wholes = list(self._wholes.get(focus_id, ()))
        return sorted(wholes, key=lambda w: (
            len(self._by_whole.get(w, ())), self._tax.label(w)))

    def _whole_candidates(self, focus_id: str) -> list[str]:
        """Co-members of the containing wholes, widening only as needed.

        Starts at the narrowest whole and widens one step at a time, because a
        distractor from ``musculature of head`` is a better item than one from
        ``head`` and far better than one from ``human body``.  Widening stops
        as soon as there are enough candidates for a five-option item.
        """
        out: list[str] = []
        seen: set[str] = set()
        for whole in self._ordered_wholes(focus_id):
            members = self._by_whole.get(whole, ())
            if len(members) > self._WHOLE_MAX_MEMBERS:
                continue
            for member in members:
                if member not in seen:
                    seen.add(member)
                    out.append(member)
            if len(out) >= self._WHOLE_TARGET:
                break
        return out

    def _shared_whole(self, focus_id: str, mesh_id: str) -> str:
        """The narrowest whole that BOTH structures are actually parts of.

        Not the focus structure's own narrowest whole: :meth:`_whole_candidates`
        widens outward when the narrowest whole has too few co-members, so a
        distractor is often a co-member of a broader whole only.  Naming the
        narrowest whole regardless would put a false containment claim in the
        provenance -- "deep part of right masseter and right frontalis are both
        parts of occipitofrontalis" -- which is exactly the class of
        plausible-but-wrong statement this package exists to prevent.
        """
        mine = self._wholes.get(focus_id, ())
        theirs = set(self._wholes.get(mesh_id, ()))
        shared = [w for w in mine if w in theirs]
        return self._tax.most_specific(shared) if shared else ""

    def _reference(self, role: str, focus_id: str, mesh_id: str) -> str:
        if role == "shares_whole":
            whole = self._shared_whole(focus_id, mesh_id)
            if not whole:
                return (f"{mesh_id} was drawn from the containing wholes of "
                        f"{focus_id}, but the tables list no whole common to "
                        f"both")
            return (f"{mesh_id} and {focus_id} are both parts of "
                    f"{whole} ({self._tax.label(whole)})")
        if role == "is_a_sibling":
            return f"{mesh_id} shares FMA superclass {self._tax.is_a_parent(focus_id)}"
        if role == "is_a_cousin":
            chain = self._tax.is_a_chain(focus_id, limit=2)
            grandparent = chain[1] if len(chain) > 1 else "?"
            return f"{mesh_id} shares FMA ancestor {grandparent}"
        meta = self._fma.get(focus_id) or {}
        if role == "same_system_and_category":
            return (f"{mesh_id} system={meta.get('system')} "
                    f"category={meta.get('category')}")
        if role == "same_system":
            return f"{mesh_id} system={meta.get('system')}"
        return f"{mesh_id} category={meta.get('category')}"

    # -- reporting ---------------------------------------------------------

    def rung_sizes(self, focus_id: str) -> dict[str, int]:
        """How many acceptable candidates each rung offers.  For diagnostics."""
        used = {self.label(focus_id).strip().lower()}
        return {
            role: len({c for c in self._candidates(role, focus_id)
                       if self._acceptable(c, focus_id, used)})
            for role, _ in LADDER
        }
