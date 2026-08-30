"""Exam item generators.  Every fact comes from data; none is authored here.

What each generator is allowed to assert, and on what basis:

===========================  ============  ==================================
Generator                    Level         Claim it makes
===========================  ============  ==================================
:meth:`identification`       L1            "this structure's name is X"
                                           -- ``fma_labels.json``
:meth:`system_of`            L2            "X belongs to the N system"
                                           -- ``fma_labels.json`` ``system``
:meth:`laterality`           L2            "the tagged structure is the
                                           left/right one" -- the FMA
                                           preferred term's own side word
:meth:`is_a`                 L3            "X is classified as a type of Y"
                                           -- FMA ``Parent FMAID`` (subClassOf)
:meth:`part_of`              L3            "X is a part of Y"
                                           -- ``conventional_part_of.txt`` and
                                           ``composite_parts.txt``
:meth:`not_part_of`          L3            "Z is not a part of Y" -- absence
                                           from both part-of tables
===========================  ============  ==================================

Two rules are enforced in code rather than trusted:

*Never state a relation the data does not carry.*  A generator returns ``None``
when its inputs are missing, so a structure with no part-of edge simply yields
no part-of question.

*Never let a distractor be true.*  For a relation question the exclusion set is
the transitive one -- every is-a ancestor and every part-of/composite whole,
not merely the option chosen as the answer -- because an option that happens to
also be a correct whole makes the item unanswerable and teaches the learner
that a true statement is false.  :meth:`_forbidden` builds that set and the
tests assert on it.

L5 (clinical reasoning) ships as a schema and a loader only.  Vignette content
requires facts this dataset does not contain (innervation, blood supply,
actions, presentations), so :func:`load_vignettes` reads externally authored
items and rejects any without a citation.  No vignette content is included.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from faceforge.anatomy.exam_items import (
    ExamItem,
    Option,
    Provenance,
)

logger = logging.getLogger(__name__)

LABELS_FILE = "assets/config/fma_labels.json"
TAXONOMY_FILE = "assets/config/fma_taxonomy.json"

#: Words the FMA preferred term uses for laterality.  Sidedness is read off
#: the standard term, never inferred from geometry.
_SIDE_WORDS = {"left": "left", "right": "right"}


class ItemGenerator:
    """Builds :class:`ExamItem` objects from the project's anatomical data.

    Parameters
    ----------
    fma:
        The crosswalk.  Loaded from the project asset when omitted.
    taxonomy:
        A :class:`~faceforge.anatomy.fma_taxonomy.Taxonomy`.
    pool:
        A :class:`~faceforge.anatomy.distractors.DistractorPool`.  Built over
        ``fma``/``taxonomy`` when omitted.
    """

    def __init__(self, fma: Optional[dict] = None, taxonomy=None, pool=None):
        if fma is None:
            from faceforge.loaders.stl_batch_loader import load_fma_labels
            fma = load_fma_labels()
        if taxonomy is None:
            from faceforge.anatomy.fma_taxonomy import get_taxonomy
            taxonomy = get_taxonomy()
        if pool is None:
            from faceforge.anatomy.distractors import DistractorPool
            pool = DistractorPool(fma=fma, taxonomy=taxonomy)
        self._fma = fma or {}
        self._tax = taxonomy
        self._pool = pool

        self._systems = sorted({
            str(v.get("system") or "") for v in self._fma.values()
        } - {""})
        # whole id -> member structure ids, for not_part_of.
        self._by_whole: dict[str, list[str]] = {}
        for mesh_id in sorted(self._fma):
            for whole in self._containing(mesh_id):
                self._by_whole.setdefault(whole, []).append(mesh_id)

    # -- helpers -----------------------------------------------------------

    def label(self, mesh_id: str) -> str:
        return self._pool.label(mesh_id)

    def _meta(self, mesh_id: str) -> dict:
        return self._fma.get(mesh_id) or {}

    def _containing(self, mesh_id: str) -> list[str]:
        """Narrowest-first containing wholes from both part-of tables."""
        seen: list[str] = []
        for w in list(self._tax.part_of(mesh_id)) + list(self._tax.wholes(mesh_id)):
            if w not in seen:
                seen.append(w)
        return seen

    def _forbidden(self, mesh_id: str) -> set[str]:
        """Everything that is genuinely true of ``mesh_id`` as a superstructure.

        A relation distractor drawn from this set would be a correct answer, so
        the set is subtracted from every candidate list.  It is deliberately
        the *unfiltered* union: the strict filters exist to choose a good
        answer, not to decide what is safe to offer as wrong.
        """
        out = {mesh_id}
        out.update(self._tax.is_a_chain(mesh_id))
        out.update(self._tax.part_of(mesh_id, strict=False))
        out.update(self._tax.wholes(mesh_id, strict=False))
        return out

    def _label_prov(self, mesh_id: str) -> Provenance:
        meta = self._meta(mesh_id)
        return Provenance(
            kind="fma_label",
            reference=mesh_id,
            detail=(f"preferred_label={meta.get('preferred_label')!r} "
                    f"system={meta.get('system')!r} "
                    f"category={meta.get('category')!r} in {LABELS_FILE}"),
        )

    def _shuffle(self, correct: Option, distractors: Sequence[Option],
                 seed: int, uid_salt: str) -> tuple[tuple[Option, ...], int]:
        """Interleave the answer among distractors, deterministically."""
        options = [correct, *distractors]
        rng = random.Random(f"{uid_salt}|{seed}")
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = tuple(options[i] for i in order)
        return shuffled, order.index(0)

    def _structure_distractors(self, focus_id: str, count: int, seed: int,
                               exclude: Iterable[str] = ()) -> list[Option]:
        out = []
        for d in self._pool.choose(focus_id, count, seed=seed, exclude=exclude):
            out.append(Option(
                text=d.label, item_id=d.item_id, role=d.role,
                provenance=(Provenance(kind=d.provenance_kind,
                                       reference=d.provenance_reference), ),
            ))
        return out

    # -- L1: identification ------------------------------------------------

    def identification(self, focus_id: str, options: int = 5, seed: int = 0,
                       fmt: str = "spot", seconds: float = 0.0
                       ) -> Optional[ExamItem]:
        """Name the indicated structure.

        The 3D render (or a scan, for L4) supplies the stimulus; the option
        list is the FMA preferred term of the focus structure against
        anatomically adjacent structures.
        """
        label = self.label(focus_id)
        if not label:
            return None
        distractors = self._structure_distractors(focus_id, options - 1, seed)
        if not distractors:
            return None
        correct = Option(text=label, item_id=focus_id, role="answer",
                         provenance=(self._label_prov(focus_id), ))
        opts, answer_index = self._shuffle(correct, distractors, seed,
                                           f"L1|{focus_id}")
        return ExamItem(
            level="L1", fmt=fmt,
            stem="Identify the indicated structure.",
            options=opts, answer_index=answer_index, focus_id=focus_id,
            provenance=(self._label_prov(focus_id), ),
            tags=("identification", ), seconds=seconds,
        )

    # -- L2: systems and laterality ---------------------------------------

    def system_of(self, focus_id: str, options: int = 5, seed: int = 0,
                  seconds: float = 0.0) -> Optional[ExamItem]:
        """To which body system does this structure belong?

        The answer is the crosswalk's ``system`` field, which was derived by
        walking the FMA parent chain; the distractors are other systems that
        actually occur in the dataset.
        """
        meta = self._meta(focus_id)
        system = str(meta.get("system") or "")
        label = self.label(focus_id)
        if not system or not label:
            return None
        others = [s for s in self._systems if s != system]
        if not others:
            return None
        rng = random.Random(f"L2sys|{focus_id}|{seed}")
        picked = rng.sample(sorted(others), min(options - 1, len(others)))
        correct = Option(text=f"{system.capitalize()} system", role="answer",
                         provenance=(self._label_prov(focus_id), ))
        distractors = [
            Option(text=f"{s.capitalize()} system", role="other_system",
                   provenance=(Provenance(
                       kind="fma_label", reference=LABELS_FILE,
                       detail=f"{s!r} occurs as a system in the crosswalk"), ))
            for s in picked
        ]
        opts, answer_index = self._shuffle(correct, distractors, seed,
                                           f"L2sys|{focus_id}")
        return ExamItem(
            level="L2", fmt="sba",
            stem=f"To which body system does the {label.lower()} belong?",
            options=opts, answer_index=answer_index, focus_id=focus_id,
            provenance=(self._label_prov(focus_id), ),
            tags=("system", ), seconds=seconds,
        )

    def laterality(self, focus_id: str, seed: int = 0, seconds: float = 0.0
                   ) -> Optional[ExamItem]:
        """Is the indicated structure the left or the right one?

        Only generated for structures whose FMA preferred term carries a side
        word, and the answer is that word.  Sidedness is never inferred from
        mesh coordinates: the render can be mirrored, the standard term cannot.
        """
        label = self.label(focus_id)
        tokens = label.lower().split()
        side = next((_SIDE_WORDS[t] for t in tokens if t in _SIDE_WORDS), "")
        if not side:
            return None
        base = " ".join(t for t in label.split() if t.lower() not in _SIDE_WORDS)
        correct = Option(text=side.capitalize(), role="answer",
                         provenance=(self._label_prov(focus_id), ))
        other = "right" if side == "left" else "left"
        distractor = Option(
            text=other.capitalize(), role="opposite_side",
            provenance=(Provenance(
                kind="fma_label", reference=focus_id,
                detail=f"preferred term says {side!r}, so {other!r} is false"), ))
        opts, answer_index = self._shuffle(correct, [distractor], seed,
                                           f"L2lat|{focus_id}")
        return ExamItem(
            level="L2", fmt="spot",
            stem=f"The indicated structure is the {base.lower()}. Which side?",
            options=opts, answer_index=answer_index, focus_id=focus_id,
            provenance=(self._label_prov(focus_id), ),
            tags=("laterality", ), seconds=seconds,
        )

    # -- L3: hierarchical relations ----------------------------------------

    def is_a(self, focus_id: str, options: int = 5, seed: int = 0,
             seconds: float = 0.0) -> Optional[ExamItem]:
        """Classification question, from the FMA subClassOf edge.

        Phrased as classification, not containment: the FMA parent edge means
        "is a kind of", and asking it as "is part of" would be a false claim.
        Distractors are the *superclass's* siblings, i.e. other classes at the
        same level of the hierarchy.
        """
        label = self.label(focus_id)
        parent = self._tax.is_a_parent(focus_id)
        parent_label = self._tax.label(parent)
        if not (label and parent and parent_label):
            return None

        forbidden = self._forbidden(focus_id)
        candidates = [
            c for c in self._tax.siblings(parent)
            if c not in forbidden and self._tax.label(c)
            and self._tax.label(c).strip().lower() != parent_label.strip().lower()
        ]
        if len(candidates) < 1:
            return None
        rng = random.Random(f"L3isa|{focus_id}|{seed}")
        picked = rng.sample(sorted(set(candidates)),
                            min(options - 1, len(set(candidates))))
        correct = Option(
            text=parent_label, item_id=parent, role="answer",
            provenance=(Provenance(
                kind="fma_is_a", reference=f"{focus_id} -> {parent}",
                detail=f"Parent FMAID edge in {TAXONOMY_FILE}"), ))
        distractors = [
            Option(text=self._tax.label(c), item_id=c, role="parent_sibling",
                   provenance=(Provenance(
                       kind="fma_is_a", reference=c,
                       detail=f"co-subclass of {self._tax.is_a_parent(parent)}, "
                              f"not an ancestor of {focus_id}"), ))
            for c in picked
        ]
        opts, answer_index = self._shuffle(correct, distractors, seed,
                                           f"L3isa|{focus_id}")
        return ExamItem(
            level="L3", fmt="sba",
            stem=(f"In the Foundational Model of Anatomy, the "
                  f"{label.lower()} is classified as a kind of which of the "
                  f"following?"),
            options=opts, answer_index=answer_index, focus_id=focus_id,
            provenance=(Provenance(
                kind="fma_is_a", reference=f"{focus_id} -> {parent}",
                detail="subClassOf edge"), ),
            tags=("is_a", "classification"), seconds=seconds,
        )

    def part_of(self, focus_id: str, options: int = 5, seed: int = 0,
                seconds: float = 0.0) -> Optional[ExamItem]:
        """Containment question, from the two part-of tables.

        The answer is the *narrowest* whole the data places the structure in;
        distractors are narrowest-wholes of other structures that are not
        wholes of this one at any level.
        """
        label = self.label(focus_id)
        wholes = self._containing(focus_id)
        answer_id = self._tax.most_specific(wholes)
        answer_label = self._tax.label(answer_id)
        if not (label and answer_id and answer_label):
            return None

        forbidden = self._forbidden(focus_id)
        forbidden_labels = {answer_label.strip().lower()}
        # Tier the candidate wholes by adjacency to the correct one, so the
        # item tests regional knowledge rather than gross elimination: a
        # "frontal bone" stem offering *interventricular septum* is answerable
        # without knowing any cranial anatomy.
        #   tier 0  other wholes sharing the answer's is-a superclass
        #           (neurocranium vs viscerocranium)
        #   tier 1  wholes that contain one of the focus structure's own is-a
        #           siblings (the region its neighbours live in)
        #   tier 2  any other whole in the dataset
        answer_parent = self._tax.is_a_parent(answer_id)
        sibling_wholes: set[str] = set()
        for sib in self._tax.siblings(focus_id)[:24]:
            sibling_wholes.update(self._containing(sib))

        tiers: dict[int, list[str]] = {0: [], 1: [], 2: []}
        for whole, _members in sorted(self._by_whole.items(),
                                      key=lambda kv: (len(kv[1]), kv[0])):
            wl = self._tax.label(whole)
            if (whole in forbidden or not wl
                    or wl.strip().lower() in forbidden_labels):
                continue
            forbidden_labels.add(wl.strip().lower())
            if answer_parent and self._tax.is_a_parent(whole) == answer_parent:
                tiers[0].append(whole)
            elif whole in sibling_wholes:
                tiers[1].append(whole)
            else:
                tiers[2].append(whole)
        rng = random.Random(f"L3part|{focus_id}|{seed}")
        picked: list[str] = []
        for tier in (0, 1, 2):
            if len(picked) >= options - 1:
                break
            pool = sorted(set(tiers[tier]) - set(picked))
            if pool:
                picked.extend(rng.sample(
                    pool, min(options - 1 - len(picked), len(pool))))
        if not picked:
            return None

        correct = Option(
            text=answer_label, item_id=answer_id, role="answer",
            provenance=(Provenance(
                kind="fma_part_of", reference=f"{focus_id} part-of {answer_id}",
                detail=f"narrowest whole in {TAXONOMY_FILE}"), ))
        distractors = [
            Option(text=self._tax.label(c), item_id=c, role="other_whole",
                   provenance=(Provenance(
                       kind="fma_part_of", reference=c,
                       detail=(f"a containing whole of other structures; not "
                               f"listed as a whole of {focus_id} at any "
                               f"level")), ))
            for c in picked
        ]
        opts, answer_index = self._shuffle(correct, distractors, seed,
                                           f"L3part|{focus_id}")
        return ExamItem(
            level="L3", fmt="sba",
            stem=f"Of which of the following is the {label.lower()} a part?",
            options=opts, answer_index=answer_index, focus_id=focus_id,
            provenance=(Provenance(
                kind="fma_part_of", reference=f"{focus_id} -> {answer_id}",
                detail="conventional part-of / composite-of tables"), ),
            tags=("part_of", ), seconds=seconds,
        )

    def not_part_of(self, whole_id: str, options: int = 5, seed: int = 0,
                    seconds: float = 0.0) -> Optional[ExamItem]:
        """"Which of the following is NOT a part of X?"

        The four true options are members of ``whole_id`` per the part-of
        tables; the answer is a structure those tables do not place inside it
        at any level and which is not one of its is-a descendants.
        """
        whole_label = self._tax.label(whole_id)
        members = self._by_whole.get(whole_id, [])
        if not whole_label or len(members) < options - 1:
            return None

        inside = set(members) | {whole_id}
        inside.update(self._tax.descendants_of(whole_id, limit=512))
        outside = [
            m for m in sorted(self._fma)
            if m not in inside and self.label(m)
            and whole_id not in self._containing(m)
        ]
        if not outside:
            return None

        # Prefer a near-miss answer.  "Which of these is NOT part of the set of
        # thoracic vertebrae?" is a real question when the answer is a lumbar
        # vertebra and a giveaway when it is the pulmonary valve.  Adjacency is
        # measured on the is-a graph relative to the true members.
        near: set[str] = set()
        for member in members[:24]:
            near.update(self._tax.siblings(member))
            near.update(self._tax.cousins(member))
        rng = random.Random(f"L3not|{whole_id}|{seed}")
        preferred = sorted(set(outside) & near)
        answer_id = rng.choice(preferred or outside)
        used_labels = {self.label(answer_id).strip().lower()}
        true_pool = []
        for m in sorted(members):
            lab = self.label(m).strip().lower()
            if lab and lab not in used_labels:
                used_labels.add(lab)
                true_pool.append(m)
        if len(true_pool) < options - 1:
            return None
        picked = rng.sample(true_pool, options - 1)

        correct = Option(
            text=self.label(answer_id), item_id=answer_id, role="answer",
            provenance=(Provenance(
                kind="fma_part_of", reference=answer_id,
                detail=(f"absent from the part-of and composite tables for "
                        f"{whole_id} and not one of its subclasses")), ))
        distractors = [
            Option(text=self.label(m), item_id=m, role="true_part",
                   provenance=(Provenance(
                       kind="fma_part_of", reference=f"{m} part-of {whole_id}",
                       detail=f"listed in {TAXONOMY_FILE}"), ))
            for m in picked
        ]
        opts, answer_index = self._shuffle(correct, distractors, seed,
                                           f"L3not|{whole_id}")
        return ExamItem(
            level="L3", fmt="sba",
            stem=(f"Which of the following is NOT a part of the "
                  f"{whole_label.lower()}?"),
            options=opts, answer_index=answer_index, focus_id=answer_id,
            provenance=(Provenance(
                kind="fma_part_of", reference=whole_id,
                detail=f"membership of {whole_label!r}"), ),
            tags=("part_of", "negative"), seconds=seconds,
        )

    # -- extended matching -------------------------------------------------

    def extended_matching(self, focus_ids: Sequence[str], seed: int = 0,
                          max_options: int = 8, seconds: float = 0.0
                          ) -> list[ExamItem]:
        """One shared option list, several stems -- the EMQ format.

        Each stem is the FMA classification of one structure ("Classified in
        the FMA as a kind of *thoracic vertebra*"), and a stem is only emitted
        when it identifies **exactly one** option in the shared list.
        Uniqueness is checked against the list rather than assumed: with
        ``first``..``twelfth thoracic vertebra`` all sharing one superclass,
        an unchecked EMQ would have twelve stems with the same answer and no
        defensible key.

        Returns items in the shared-option format (``fmt="emq"``); an empty
        list when fewer than two stems survive the uniqueness check, because a
        one-stem EMQ is just a badly formatted single-best-answer item.
        """
        labelled = [(f, self.label(f)) for f in focus_ids]
        labelled = [(f, lab) for f, lab in labelled if lab][:max_options]
        if len(labelled) < 3:
            return []

        options = tuple(
            Option(text=lab, item_id=f, role="emq_option",
                   provenance=(self._label_prov(f), ))
            for f, lab in labelled
        )
        parents = {f: self._tax.is_a_parent(f) for f, _ in labelled}
        by_parent: dict[str, list[str]] = {}
        for f, parent in parents.items():
            if parent:
                by_parent.setdefault(parent, []).append(f)

        out: list[ExamItem] = []
        for index, (focus_id, _) in enumerate(labelled):
            parent = parents.get(focus_id, "")
            parent_label = self._tax.label(parent)
            if not parent or not parent_label:
                continue
            if len(by_parent.get(parent, ())) != 1:
                continue          # stem does not discriminate; drop it
            out.append(ExamItem(
                level="L3", fmt="emq",
                stem=(f"Classified in the Foundational Model of Anatomy as a "
                      f"kind of {parent_label.lower()}."),
                options=options, answer_index=index, focus_id=focus_id,
                provenance=(Provenance(
                    kind="fma_is_a", reference=f"{focus_id} -> {parent}",
                    detail=(f"unique among the {len(options)} options in this "
                            f"matching set")), ),
                tags=("emq", "is_a"), seconds=seconds,
            ))
        return out if len(out) >= 2 else []

    # -- batch -------------------------------------------------------------

    #: Which generators serve which level.
    BY_LEVEL = {
        "L1": ("identification", ),
        "L2": ("system_of", "laterality"),
        "L3": ("is_a", "part_of"),
    }

    def generate(self, level: str, focus_ids: Sequence[str], count: int,
                 options: int = 5, seed: int = 0, seconds: float = 0.0
                 ) -> list[ExamItem]:
        """Up to ``count`` items of ``level`` over ``focus_ids``.

        Focus structures are taken in the order given -- the caller has already
        ordered them (by spaced repetition, or by curriculum tier) -- and each
        generator for the level is tried in turn.  A structure the data cannot
        support at this level is skipped silently; the shortfall is visible in
        the returned length.
        """
        names = self.BY_LEVEL.get(level, ())
        if not names:
            return []
        out: list[ExamItem] = []
        seen_uids: set[str] = set()
        for focus_id in focus_ids:
            for name in names:
                if len(out) >= count:
                    return out
                item = getattr(self, name)(focus_id, seed=seed, seconds=seconds) \
                    if name == "laterality" else \
                    getattr(self, name)(focus_id, options=options, seed=seed,
                                        seconds=seconds)
                if item is not None and item.uid not in seen_uids:
                    seen_uids.add(item.uid)
                    out.append(item)
        return out

    def coverage(self, focus_ids: Iterable[str]) -> dict[str, int]:
        """How many of ``focus_ids`` each generator can actually serve.

        Reported rather than assumed: this is the honest answer to "how much of
        the dataset supports each question type".
        """
        counts = {name: 0 for names in self.BY_LEVEL.values() for name in names}
        for focus_id in focus_ids:
            for name in counts:
                if getattr(self, name)(focus_id, seed=0) is not None:
                    counts[name] += 1
        return counts


# -- L5: clinical vignettes, schema and loader only -----------------------

VIGNETTE_SCHEMA = {
    "required": ("uid", "stem", "options", "answer_index", "citation"),
    "optional": ("focus_id", "level", "seconds", "tags", "explanation"),
    "notes": (
        "A vignette asserts facts this dataset does not contain -- "
        "innervation, blood supply, action, presentation, management -- so it "
        "must come with a citation to a source a reviewer can check "
        "(textbook edition and page, or a DOI). Items without one are "
        "rejected by load_vignettes()."
    ),
}


@dataclass(frozen=True)
class VignetteRejection:
    """Why one authored item was not loaded."""

    uid: str
    reasons: tuple[str, ...]


def load_vignettes(path: Path | str) -> tuple[list[ExamItem], list[VignetteRejection]]:
    """Load externally authored L5 items, rejecting any without a citation.

    Returns ``(items, rejections)``.  This project ships no vignette content:
    the format exists so a school can supply its own, and the loader is the
    place the citation requirement is enforced.  An item that passes here is
    still subject to :func:`faceforge.anatomy.exam_items.present`, which will
    refuse it in exam mode if its provenance is not authored-with-citation.
    """
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], []
    except (OSError, ValueError) as exc:
        logger.error("Vignette file %s could not be read: %s", target, exc)
        return [], [VignetteRejection(uid=str(target), reasons=("unreadable", ))]

    rows = payload.get("items", payload if isinstance(payload, list) else [])
    items: list[ExamItem] = []
    rejected: list[VignetteRejection] = []
    for row in rows:
        uid = str(row.get("uid", "")) if isinstance(row, dict) else ""
        reasons = _vignette_problems(row)
        if reasons:
            rejected.append(VignetteRejection(uid=uid or "<no uid>",
                                              reasons=tuple(reasons)))
            continue
        citation = str(row["citation"]).strip()
        options = tuple(
            Option(text=str(o), role="authored",
                   provenance=(Provenance(kind="citation", reference=citation), ))
            for o in row["options"]
        )
        items.append(ExamItem(
            level="L5", fmt="sba", stem=str(row["stem"]),
            options=options, answer_index=int(row["answer_index"]),
            focus_id=str(row.get("focus_id", "")),
            provenance=(Provenance(kind="citation", reference=citation,
                                   detail="externally authored vignette"), ),
            citation=citation,
            tags=tuple(row.get("tags", ())) or ("vignette", ),
            seconds=float(row.get("seconds", 0.0)),
            uid=uid or "",
        ))
    return items, rejected


def _vignette_problems(row: object) -> list[str]:
    if not isinstance(row, dict):
        return ["not an object"]
    out = [f"missing {key!r}" for key in VIGNETTE_SCHEMA["required"]
           if key not in row]
    if out:
        return out
    if not str(row.get("citation", "")).strip():
        out.append("empty citation")
    options = row.get("options")
    if not isinstance(options, list) or len(options) < 2:
        out.append("needs at least two options")
    elif not isinstance(row.get("answer_index"), int) or \
            not 0 <= row["answer_index"] < len(options):
        out.append("answer_index out of range")
    if not str(row.get("stem", "")).strip():
        out.append("empty stem")
    return out
