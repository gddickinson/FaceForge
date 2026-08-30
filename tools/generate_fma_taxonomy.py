"""Generate assets/config/fma_taxonomy.json from the BodyParts3D FMA table.

The FMA table shipped with BodyParts3D
(``BodyParts3D/assets/BodyParts3D_data/FMA.csv``, 104,723 data rows) has three
columns: ``FMAID``, ``Preferred Label``, ``Parent FMAID``.  That parent edge is
the only ontological structure this project has access to, and it is what makes
hierarchical exam questions ("of what is the frontal bone a part?", "which of
these is NOT part of the mandible?") derivable from data instead of authored.

The whole table is 5.9 MB and lives outside this repository, so it is not a
runtime dependency.  This script extracts the *closure* needed for question
generation and writes it into ``assets/config``:

* every structure in ``fma_labels.json`` (the 923 meshes this app can load),
* every ancestor of those structures, up to the FMA root,
* every child of every one of those ancestors -- required so that *siblings*
  (nodes sharing a parent) can be enumerated for distractor generation.

Run from the project root::

    python -m tools.generate_fma_taxonomy \\
        --fma-csv /path/to/BodyParts3D_data/FMA.csv

Idempotent: given the same inputs it writes byte-identical output (keys sorted,
fixed separators).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_FMA_CSV = Path(
    "/Users/george/Documents/GitHub/BodyParts3D/assets/BodyParts3D_data/FMA.csv"
)
SCHEMA_VERSION = 1


def read_fma_csv(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(label_by_id, parent_by_id)`` keyed by bare numeric FMA id."""
    labels: dict[str, str] = {}
    parents: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        missing = {"FMAID", "Preferred Label", "Parent FMAID"} - set(
            reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            fid = (row["FMAID"] or "").strip()
            if not fid:
                continue
            labels[fid] = (row["Preferred Label"] or "").strip()
            parent = (row["Parent FMAID"] or "").strip()
            # Top-level FMA classes carry the OWL root URI
            # ("http://www.w3.org/2002/07/owl#Thing") rather than an id; those
            # are roots, so they simply get no parent edge.
            if parent and parent != fid and parent.isdigit():
                parents[fid] = parent
    return labels, parents


def bare_id(mesh_id: str) -> str:
    """``"FMA52734"`` / ``"FMA14543nsn"`` -> ``"52734"`` / ``"14543"``.

    The BodyParts3D mesh ids carry an ``FMA`` prefix and sometimes an ``nsn``
    suffix ("no standard name"); the FMA table keys are bare integers.
    """
    digits = "".join(ch for ch in mesh_id if ch.isdigit())
    return digits


def read_tsv_pairs(path: Path) -> list[tuple[str, str, str, str]]:
    """Rows of a BodyParts3D relation TSV as ``(id, name, other_id, other)``.

    Both relation files shipped with the dataset have the same shape: a header
    line then four tab-separated columns.  Ids carry the ``FMA``/``BP`` prefix
    here (unlike FMA.csv), and ``BP`` ids are BodyParts3D-local composites with
    no FMA identity.
    """
    out: list[tuple[str, str, str, str]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)                              # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            out.append((parts[0].strip().strip('"'), parts[1].strip(),
                        parts[2].strip().strip('"'), parts[3].strip()))
    return out


def build(fma_csv: Path, labels_json: Path,
          part_of_txt: Path | None = None,
          composite_txt: Path | None = None) -> dict:
    labels, parents = read_fma_csv(fma_csv)
    structures = json.loads(labels_json.read_text(encoding="utf-8"))["structures"]

    seeds = {bare_id(k) for k in structures}
    seeds.discard("")

    # 1. ancestors of every seed
    closure: set[str] = set()
    for seed in seeds:
        node = seed
        seen_chain = set()
        while node and node not in seen_chain:
            seen_chain.add(node)
            closure.add(node)
            node = parents.get(node, "")

    # 2. children of every node in the closure (for sibling enumeration)
    children: dict[str, list[str]] = {}
    for node, parent in parents.items():
        children.setdefault(parent, []).append(node)
    with_children = set(closure)
    for node in closure:
        with_children.update(children.get(node, ()))

    nodes = {
        fid: {"label": labels.get(fid, ""), "parent": parents.get(fid, "")}
        for fid in sorted(with_children, key=int)
    }

    # 3. genuine part-of edges.  These are a DIFFERENT relation from the
    #    FMA.csv parent edge, which is subClassOf (is-a): "frontal bone is-a
    #    flat bone", not "frontal bone part-of flat bone".  Conflating them
    #    would put a false anatomical claim in front of a learner, so they are
    #    stored separately and the question generators say which they used.
    part_of: dict[str, list[str]] = {}
    part_labels: dict[str, str] = {}
    n_part_rows = 0
    if part_of_txt is not None and part_of_txt.exists():
        for whole_id, whole_name, part_id, part_name in read_tsv_pairs(part_of_txt):
            n_part_rows += 1
            part_labels.setdefault(part_id, part_name)
            part_labels.setdefault(whole_id, whole_name)
            part_of.setdefault(part_id, [])
            if whole_id not in part_of[part_id]:
                part_of[part_id].append(whole_id)

    # 4. composite -> primitive.  BodyParts3D splits some structures into a
    #    composite (unsided/whole) and its primitives (left/right, segments).
    #    This is the whole/part relation the meshes themselves are organised
    #    by, and it is what makes "of which structure is the left first rib a
    #    part?" answerable.
    composite_of: dict[str, list[str]] = {}
    n_comp_rows = 0
    if composite_txt is not None and composite_txt.exists():
        for comp_id, comp_name, prim_id, prim_name in read_tsv_pairs(composite_txt):
            n_comp_rows += 1
            part_labels.setdefault(comp_id, comp_name)
            part_labels.setdefault(prim_id, prim_name)
            composite_of.setdefault(prim_id, [])
            if comp_id not in composite_of[prim_id]:
                composite_of[prim_id].append(comp_id)

    return {
        "schema_version": SCHEMA_VERSION,
        "_comment": (
            "Three DISTINCT anatomical relations, kept separate on purpose. "
            "'nodes' carries the FMA.csv parent edge, which is subClassOf "
            "(is-a) -- 'frontal bone is-a flat bone'. 'part_of' carries the "
            "conventional part-of relation from conventional_part_of.txt. "
            "'composite_of' carries the BodyParts3D composite->primitive "
            "relation from composite_parts.txt (a whole and its sided or "
            "segmental parts). Node coverage is every structure this app can "
            "load, plus their is-a ancestors, plus those ancestors' children "
            "(needed to enumerate siblings for exam distractors). Generated "
            "by tools/generate_fma_taxonomy.py; do not hand-edit."
        ),
        "_source": str(fma_csv),
        "_source_part_of": str(part_of_txt) if part_of_txt else "",
        "_source_composite": str(composite_txt) if composite_txt else "",
        "_seed_structures": len(seeds),
        "_generated_nodes": len(nodes),
        "_part_of_rows": n_part_rows,
        "_composite_rows": n_comp_rows,
        "nodes": nodes,
        "labels": {k: v for k, v in sorted(part_labels.items()) if v},
        "part_of": {k: v for k, v in sorted(part_of.items()) if v},
        "composite_of": {k: v for k, v in sorted(composite_of.items()) if v},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fma-csv", type=Path, default=DEFAULT_FMA_CSV)
    ap.add_argument("--labels", type=Path,
                    default=Path("assets/config/fma_labels.json"))
    ap.add_argument("--part-of", type=Path,
                    default=DEFAULT_FMA_CSV.parent / "conventional_part_of.txt")
    ap.add_argument("--composite", type=Path,
                    default=DEFAULT_FMA_CSV.parent / "composite_parts.txt")
    ap.add_argument("--out", type=Path,
                    default=Path("assets/config/fma_taxonomy.json"))
    args = ap.parse_args(argv)

    if not args.fma_csv.exists():
        print(f"FMA table not found at {args.fma_csv}", file=sys.stderr)
        return 2

    payload = build(args.fma_csv, args.labels, args.part_of, args.composite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.out}: {payload['_generated_nodes']} is-a nodes, "
          f"{len(payload['part_of'])} part-of subjects, "
          f"{len(payload['composite_of'])} composite subjects, "
          f"from {payload['_seed_structures']} seed structures "
          f"({args.out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
