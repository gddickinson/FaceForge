"""Structures that belong to one sex, shown only for that sex.

The BodyParts3D set this project ships is a male cadaver, and eight of the
configured organs are male reproductive structures: the testes, the
epididymides, the seminal vesicles, the prostate and the glans.  There are no
female equivalents in the asset set -- no uterus, no ovary, no vagina -- so a
female model cannot be given the right organs, but it certainly should not be
left with the wrong ones.

They fade out across the first half of the slider and are gone by the middle
of it, so the change happens where the body is still ambiguous rather than
snapping at either end.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Structures a female model must not have.  Matched on the mesh name exactly,
#: because "Prostate" is a whole name and a substring test would also catch
#: "Prostatic Urethra", which is not a reproductive organ.
MALE_ONLY: frozenset[str] = frozenset({
    "Right Testis", "Left Testis",
    "Right Epididymis", "Left Epididymis",
    "R Seminal Vesicle", "L Seminal Vesicle",
    "Prostate", "Glans Penis",
    # Both sexes have a urethra, but this one is the male's: 20 cm, running
    # through the penis and clearly visible below the pubic arch on a body
    # that should not have one.  A missing urethra is an omission; a male
    # urethra on a female model is a mistake.
    "Urethra",
})

#: Gender at which they have gone entirely.
HIDDEN_BY = 0.5


def apply(root: Any, gender: float) -> int:
    """Hide the male-only structures for a female model.  Returns how many."""
    if root is None:
        return 0
    hide = float(gender) >= HIDDEN_BY
    hidden = 0
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if (node.name or "") in MALE_ONLY:
            node.visible = not hide
            hidden += int(hide)
    if hidden:
        logger.info("Male reproductive structures hidden at gender %.2f: %d",
                    gender, hidden)
    return hidden
