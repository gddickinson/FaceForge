"""Read-only access to the FMA relation graph shipped in assets/config.

Three relations, kept apart because they are not the same claim:

``is-a`` (subClassOf)
    From ``FMA.csv``'s ``Parent FMAID`` column.  *Frontal bone is-a flat bone.*
    This is a classification, not a containment: reading it as "part of" would
    put a false anatomical statement in front of a learner, which is exactly
    what this module exists to prevent.
``part-of``
    From BodyParts3D's ``conventional_part_of.txt``.  *Frontal bone is part of
    the neurocranium.*
``composite-of``
    From BodyParts3D's ``composite_parts.txt``: the composite (unsided or
    whole) structure a primitive belongs to.  *Right frontalis is a primitive
    of frontalis.*  Transitive in the source file, so :func:`Taxonomy.wholes`
    returns the whole set and callers pick specificity.

The part-of file contains some rows that duplicate the is-a parent (``frontal
bone`` -> ``flat bone`` appears in both).  :meth:`Taxonomy.part_of` therefore
subtracts the is-a ancestor set by default -- see its docstring -- so a
part-of question never asserts a classification as a containment.

All ids here are BodyParts3D mesh ids (``"FMA52734"``).  The generated file is
keyed by bare numerics for the is-a graph and by prefixed ids for the two
part-of tables, which is an artefact of the upstream files; this module hides
that.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from faceforge.constants import CONFIG_DIR

logger = logging.getLogger(__name__)

TAXONOMY_FILENAME = "fma_taxonomy.json"
SCHEMA_VERSION = 1


def bare_id(mesh_id: str) -> str:
    """``"FMA52734"`` / ``"FMA14543nsn"`` -> ``"52734"``."""
    return "".join(ch for ch in str(mesh_id) if ch.isdigit())


def prefixed_id(fma_id: str) -> str:
    """``"52734"`` -> ``"FMA52734"``; already-prefixed ids pass through."""
    s = str(fma_id)
    return s if s.upper().startswith(("FMA", "BP")) else f"FMA{s}"


@dataclass(frozen=True)
class Relation:
    """One edge, carrying which relation it is and where it came from.

    ``kind`` is one of ``"is_a"``, ``"part_of"``, ``"composite_of"``.  A
    question generator quotes ``kind`` in its stem and ``source`` in its
    provenance, so a learner (or a reviewer) can see the claim's basis.
    """

    kind: str
    subject_id: str
    object_id: str
    subject_label: str
    object_label: str
    source: str


class Taxonomy:
    """The relation graph, loaded from ``assets/config/fma_taxonomy.json``.

    A missing or unreadable file is not fatal: every accessor degrades to
    "no relation known", which makes the hierarchical question generators emit
    nothing rather than emit something invented.  :attr:`available` says which
    happened.
    """

    def __init__(self, payload: Optional[dict] = None):
        if payload is None:
            payload = _load_payload()
        self._payload = payload or {}
        self._nodes: dict[str, dict] = self._payload.get("nodes", {}) or {}
        self._labels: dict[str, str] = self._payload.get("labels", {}) or {}
        self._part_of: dict[str, list[str]] = self._payload.get("part_of", {}) or {}
        self._composite: dict[str, list[str]] = \
            self._payload.get("composite_of", {}) or {}
        # composite -> its primitives, inverted from composite_of, so
        # specificity of a whole can be measured (see most_specific).
        self._primitives: dict[str, list[str]] = {}
        for primitive, composites in self._composite.items():
            for composite in composites:
                self._primitives.setdefault(composite, []).append(primitive)
        self._children: dict[str, list[str]] = {}
        for fid, node in self._nodes.items():
            parent = node.get("parent") or ""
            if parent:
                self._children.setdefault(parent, []).append(fid)
        for kids in self._children.values():
            kids.sort(key=_int_key)
        self.source = self._payload.get("_source", "")
        self.part_of_source = self._payload.get("_source_part_of", "")
        self.composite_source = self._payload.get("_source_composite", "")

    @property
    def available(self) -> bool:
        return bool(self._nodes)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # -- labels ------------------------------------------------------------

    def label(self, mesh_id: str) -> str:
        """FMA preferred label for any id in either table ("" if unknown)."""
        node = self._nodes.get(bare_id(mesh_id))
        if node and node.get("label"):
            return node["label"]
        return self._labels.get(prefixed_id(mesh_id), "")

    # -- is-a --------------------------------------------------------------

    def is_a_parent(self, mesh_id: str) -> str:
        """Immediate superclass as a prefixed id, or ""."""
        node = self._nodes.get(bare_id(mesh_id))
        parent = (node or {}).get("parent") or ""
        return prefixed_id(parent) if parent else ""

    def is_a_chain(self, mesh_id: str, limit: int = 32) -> list[str]:
        """Superclasses from the immediate parent up to the FMA root.

        Cycle-safe (the source is not guaranteed acyclic) and depth-capped.
        """
        out: list[str] = []
        seen = {bare_id(mesh_id)}
        node = (self._nodes.get(bare_id(mesh_id)) or {}).get("parent") or ""
        while node and node not in seen and len(out) < limit:
            seen.add(node)
            out.append(prefixed_id(node))
            node = (self._nodes.get(node) or {}).get("parent") or ""
        return out

    def siblings(self, mesh_id: str) -> list[str]:
        """Other structures sharing this structure's immediate superclass.

        Deterministic (numeric id order).  Excludes ``mesh_id`` itself.
        """
        parent = bare_id(self.is_a_parent(mesh_id))
        if not parent:
            return []
        me = bare_id(mesh_id)
        return [prefixed_id(k) for k in self._children.get(parent, []) if k != me]

    def cousins(self, mesh_id: str) -> list[str]:
        """Structures sharing the *grandparent* class but not the parent.

        The fallback when a structure has too few siblings for a five-option
        item: the sibling groups in this dataset have a median size of 2.
        """
        chain = self.is_a_chain(mesh_id, limit=2)
        if len(chain) < 2:
            return []
        grandparent = bare_id(chain[1])
        exclude = {bare_id(mesh_id)} | {bare_id(s) for s in self.siblings(mesh_id)}
        out: list[str] = []
        for uncle in self._children.get(grandparent, []):
            for kid in self._children.get(uncle, []):
                if kid not in exclude:
                    out.append(prefixed_id(kid))
        return out

    def descendants_of(self, mesh_id: str, limit: int = 4096) -> list[str]:
        """All subclasses beneath ``mesh_id`` (breadth-first, deterministic)."""
        out: list[str] = []
        frontier = [bare_id(mesh_id)]
        seen = set(frontier)
        while frontier and len(out) < limit:
            node = frontier.pop(0)
            for kid in self._children.get(node, []):
                if kid in seen:
                    continue
                seen.add(kid)
                out.append(prefixed_id(kid))
                frontier.append(kid)
        return out

    # -- part-of -----------------------------------------------------------

    def part_of(self, mesh_id: str, strict: bool = True) -> list[str]:
        """Wholes that ``mesh_id`` is a conventional part of.

        With ``strict=True`` (the default) any whole that is also an is-a
        ancestor is removed.  The upstream file mixes the two relations for
        some structures -- ``frontal bone`` is listed as a part of ``flat
        bone``, which is its *superclass* -- and asserting a classification as
        a containment is precisely the kind of plausible-but-wrong fact this
        module refuses to produce.
        """
        wholes = list(self._part_of.get(prefixed_id(mesh_id), ()))
        if not strict:
            return wholes
        ancestors = {bare_id(a) for a in self.is_a_chain(mesh_id)}
        return [w for w in wholes if bare_id(w) not in ancestors]

    def wholes(self, mesh_id: str, strict: bool = True) -> list[str]:
        """Composite structures this structure is a primitive of.

        The upstream relation is transitive, so this returns everything from
        the immediate unsided parent up to ``human body``.  Use
        :meth:`most_specific` to pick one.  ``strict=True`` drops entries that
        are really the is-a superclass, for the reason given in
        :meth:`part_of`.
        """
        wholes = list(self._composite.get(prefixed_id(mesh_id), ()))
        if not strict:
            return wholes
        ancestors = {bare_id(a) for a in self.is_a_chain(mesh_id)}
        return [w for w in wholes if bare_id(w) not in ancestors]

    def primitive_count(self, mesh_id: str) -> int:
        """How many primitives this composite has, per composite_parts.txt.

        The specificity measure for composites: ``neurocranium`` has far fewer
        primitives than ``human body``, and that is a property of the data
        rather than a judgement.
        """
        return len(self._primitives.get(prefixed_id(mesh_id), ()))

    def most_specific(self, ids: Iterable[str]) -> str:
        """The narrowest of ``ids``.  Deterministic; "" for an empty input.

        Narrowness is measured on whichever table knows the id: the number of
        primitives it aggregates (composites) or the number of subclasses
        beneath it (is-a classes).  An id known to neither is treated as
        maximally broad, so it never wins by default.
        """
        candidates = [i for i in ids if i]
        if not candidates:
            return ""

        def key(i: str):
            primitives = self.primitive_count(i)
            subclasses = len(self.descendants_of(i, limit=512))
            known = primitives or subclasses
            return (0 if known else 1, primitives or subclasses or 10 ** 6,
                    _int_key(bare_id(i)))

        return min(candidates, key=key)

    # -- packaged relations ------------------------------------------------

    def relation(self, kind: str, subject_id: str, object_id: str) -> Relation:
        """Build a :class:`Relation` with labels and source filled in."""
        source = {
            "is_a": self.source,
            "part_of": self.part_of_source,
            "composite_of": self.composite_source,
        }.get(kind, "")
        return Relation(
            kind=kind,
            subject_id=prefixed_id(subject_id),
            object_id=prefixed_id(object_id),
            subject_label=self.label(subject_id),
            object_label=self.label(object_id),
            source=source or TAXONOMY_FILENAME,
        )


def _int_key(value: str) -> tuple[int, int | str]:
    """Sort bare ids numerically, non-numeric ones last but stably."""
    return (0, int(value)) if value.isdigit() else (1, value)


def _load_payload(path: Optional[Path] = None) -> dict:
    target = Path(path) if path is not None else CONFIG_DIR / TAXONOMY_FILENAME
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(
            "FMA taxonomy missing at %s; hierarchical exam questions will not "
            "be generated (run tools/generate_fma_taxonomy.py)", target)
        return {}
    except (OSError, ValueError):
        logger.exception("FMA taxonomy at %s could not be read", target)
        return {}
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        logger.error("FMA taxonomy at %s is schema %r, expected %d; not used",
                     target, version, SCHEMA_VERSION)
        return {}
    return payload


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    """Cached taxonomy over the project's generated asset."""
    return Taxonomy()
