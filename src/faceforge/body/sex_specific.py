"""Structures that belong to one sex, shown only for that sex.

The BodyParts3D set this project ships is a male cadaver, so *every* organ it
categorises as reproductive is a male one: the testes, the epididymides, the
deferent ducts, the seminal vesicles, the prostate, and the three erectile
bodies of the penis.  There are no female equivalents in the asset set -- no
uterus, no ovary, no vagina -- so a female model cannot be given the right
organs, but it certainly should not be left with the wrong ones.

The list is read from ``organs.json`` rather than written out here.  It was
written out here once, and it went stale the way a hand-kept list does: it
named the glans but neither the corpus cavernosum nor the corpus spongiosum,
so a female model lost the tip of the penis and kept the shaft.  The config
already knows which organs are reproductive; asking it is both correct today
and correct when the config gains another one.

They fade out across the first half of the slider and are gone by the middle
of it, so the change happens where the body is still ambiguous rather than
snapping at either end.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The organ category that is male-only in this asset set.
MALE_CATEGORY = "reproductive"

#: Male-only structures the category does not catch.  Matched on the mesh name
#: exactly, because "Urethra" is a whole name and a substring test would also
#: catch "Prostatic Urethra", which is not a reproductive organ.
#:
#: Both sexes have a urethra, but this one is the male's: 20 cm, running
#: through the penis and clearly visible below the pubic arch on a body that
#: should not have one.  A missing urethra is an omission; a male urethra on a
#: female model is a mistake.
NAMED_MALE_ONLY: frozenset[str] = frozenset({"Urethra"})

#: Gender at which they have gone entirely.
HIDDEN_BY = 0.5

_cache: frozenset[str] | None = None


def male_only() -> frozenset[str]:
    """Every structure a female model must not have, read from the config."""
    global _cache
    if _cache is not None:
        return _cache
    names = set(NAMED_MALE_ONLY)
    try:
        from faceforge.core.config_loader import load_config

        data = load_config("organs.json")
        items = data if isinstance(data, list) else list(data.values())
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("category", "")).lower() == MALE_CATEGORY:
                name = item.get("name")
                if name:
                    names.add(str(name))
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        # No config on disk (the fast test tier runs without assets).  The
        # named set still holds; a missing category list is not a reason to
        # stop hiding the urethra.
        logger.warning("organs.json unreadable, sex-specific list is partial: %s",
                       exc)
    _cache = frozenset(names)
    return _cache


def apply(root: Any, gender: float) -> int:
    """Hide the male-only structures for a female model.  Returns how many."""
    if root is None:
        return 0
    hide = float(gender) >= HIDDEN_BY
    names = male_only()
    hidden = 0
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if (node.name or "") in names:
            node.visible = not hide
            hidden += int(hide)
    if hidden:
        logger.info("Male reproductive structures hidden at gender %.2f: %d",
                    gender, hidden)
    return hidden
