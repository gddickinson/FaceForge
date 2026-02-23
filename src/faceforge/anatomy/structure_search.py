"""Natural language anatomical structure search.

Supports queries like "muscles that flex the elbow" by tokenising query
text and matching against a pre-built index of mesh names, joint actions,
and anatomical regions.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Joint action verbs → canonical action tokens
_ACTION_VERBS = {
    "flex": "flex", "flexes": "flex", "flexion": "flex",
    "extend": "extend", "extends": "extend", "extension": "extend",
    "abduct": "abduct", "abducts": "abduct", "abduction": "abduct",
    "adduct": "adduct", "adducts": "adduct", "adduction": "adduct",
    "rotate": "rotate", "rotates": "rotate", "rotation": "rotate",
    "elevate": "elevate", "elevates": "elevate", "elevation": "elevate",
    "depress": "depress", "depresses": "depress", "depression": "depress",
    "pronate": "pronate", "pronates": "pronate", "pronation": "pronate",
    "supinate": "supinate", "supinates": "supinate", "supination": "supinate",
    "invert": "invert", "inverts": "invert", "inversion": "invert",
    "evert": "evert", "everts": "evert", "eversion": "evert",
    "open": "open", "opens": "open",
    "close": "close", "closes": "close",
}

# Joint name synonyms
_JOINT_SYNONYMS = {
    "jaw": "jaw", "mandible": "jaw", "tmj": "jaw",
    "shoulder": "shoulder",
    "elbow": "elbow",
    "wrist": "wrist",
    "hip": "hip",
    "knee": "knee",
    "ankle": "ankle",
    "spine": "spine", "back": "spine", "trunk": "spine",
    "neck": "neck", "cervical": "neck",
    "finger": "finger", "fingers": "finger", "hand": "hand",
    "toe": "toe", "toes": "toe", "foot": "foot",
}

# Region synonyms
_REGION_SYNONYMS = {
    "head": "head", "skull": "head", "cranium": "head", "face": "head",
    "neck": "neck", "cervical": "neck",
    "thorax": "thorax", "chest": "thorax", "rib": "thorax",
    "abdomen": "abdomen", "belly": "abdomen", "stomach": "abdomen",
    "arm": "arm", "upper limb": "arm",
    "leg": "leg", "lower limb": "leg",
    "pelvis": "pelvis", "hip": "pelvis",
    "hand": "hand",
    "foot": "foot",
    "brain": "brain",
}

# Category synonyms
_CATEGORY_SYNONYMS = {
    "muscle": "muscle", "muscles": "muscle",
    "bone": "bone", "bones": "bone", "skeleton": "bone", "skeletal": "bone",
    "organ": "organ", "organs": "organ", "viscera": "organ",
    "vessel": "vessel", "vessels": "vessel", "vascular": "vessel",
    "artery": "vessel", "arteries": "vessel", "vein": "vessel", "veins": "vessel",
    "nerve": "nerve", "nerves": "nerve", "neural": "nerve",
    "ligament": "ligament", "ligaments": "ligament",
    "tendon": "tendon", "tendons": "tendon",
}


@dataclass
class SearchEntry:
    """A single searchable structure."""
    mesh_name: str
    display_name: str
    category: str = ""        # "muscle", "bone", "organ", "vessel", etc.
    region: str = ""          # "head", "neck", "thorax", "arm", "leg", etc.
    keywords: list[str] = field(default_factory=list)
    joint_actions: list[str] = field(default_factory=list)  # "flex_elbow", etc.


@dataclass
class SearchResult:
    """A search result with relevance score."""
    entry: SearchEntry
    score: float = 0.0
    match_reason: str = ""


class AnatomySearchIndex:
    """Builds and queries a searchable index of anatomical structures.

    Supports keyword matching, functional queries (e.g. "muscles that flex
    the elbow"), and category/region filtering.
    """

    def __init__(self):
        self._entries: list[SearchEntry] = []
        self._built = False

    @property
    def entries(self) -> list[SearchEntry]:
        return self._entries

    def build_from_muscle_configs(self, muscle_configs: list[dict]) -> None:
        """Add muscles from loaded config dicts.

        Each dict should have: name, originBones, insertionBones.
        """
        for cfg in muscle_configs:
            name = cfg.get("name", "")
            if not name:
                continue

            # Derive region from name suffix
            region = self._infer_region(name, cfg)

            # Derive joint actions from origin/insertion bones
            actions = self._infer_actions(cfg)

            # Build keyword list from name tokens
            keywords = self._tokenize(name)

            entry = SearchEntry(
                mesh_name=name,
                display_name=name,
                category="muscle",
                region=region,
                keywords=keywords,
                joint_actions=actions,
            )
            self._entries.append(entry)

    def build_from_names(self, names: list[str]) -> None:
        """Build index from a list of mesh names, auto-categorising each."""
        for name in names:
            keywords = self._tokenize(name)
            name_lower = name.lower()
            # Auto-detect category
            if "muscle" in name_lower or any(w in name_lower for w in
                    ("biceps", "triceps", "deltoid", "trapezius", "pectoralis")):
                category = "muscle"
            elif any(w in name_lower for w in ("bone", "skull", "mandible",
                     "vertebr", "rib", "sternum", "humerus", "femur",
                     "tibia", "fibula", "radius", "ulna", "scapula",
                     "clavicle", "pelvis", "ilium", "ischium")):
                category = "bone"
            elif any(w in name_lower for w in ("heart", "lung", "liver",
                     "kidney", "stomach", "intestin", "spleen", "pancreas",
                     "bladder", "thyroid", "adrenal")):
                category = "organ"
            elif any(w in name_lower for w in ("artery", "vein", "aorta",
                     "carotid", "jugular", "vena", "vessel")):
                category = "vessel"
            elif any(w in name_lower for w in ("nerve", "brain", "cerebr",
                     "cerebel", "spinal cord")):
                category = "nerve"
            else:
                category = ""
            region = self._infer_region_from_name(name)
            entry = SearchEntry(
                mesh_name=name,
                display_name=name,
                category=category,
                region=region,
                keywords=keywords,
            )
            self._entries.append(entry)
        self._built = True

    def add_structures(self, names: list[str], category: str = "",
                       region: str = "") -> None:
        """Add non-muscle structures (bones, organs, vessels, etc.)."""
        for name in names:
            keywords = self._tokenize(name)
            entry = SearchEntry(
                mesh_name=name,
                display_name=name,
                category=category,
                region=region or self._infer_region_from_name(name),
                keywords=keywords,
            )
            self._entries.append(entry)
        self._built = True

    def search(self, query: str) -> list[SearchResult]:
        """Search the index with a natural language query.

        Returns results sorted by relevance score (highest first).
        """
        if not query.strip():
            return []

        query_lower = query.lower().strip()
        tokens = self._tokenize(query_lower)

        # Parse functional query pattern: "muscles that [verb] the [joint]"
        func_match = self._parse_functional_query(query_lower)

        results: list[SearchResult] = []
        for entry in self._entries:
            score, reason = self._score_entry(entry, tokens, func_match)
            if score > 0:
                results.append(SearchResult(entry=entry, score=score,
                                           match_reason=reason))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:50]  # Cap at 50 results

    def _score_entry(self, entry: SearchEntry, tokens: list[str],
                     func_match: Optional[tuple[str, str, str]]
                     ) -> tuple[float, str]:
        """Score an entry against query tokens and functional match.

        Returns (score, reason) where score > 0 means a match.
        """
        score = 0.0
        reasons = []

        # Functional query match (highest priority)
        if func_match:
            cat_filter, action, joint = func_match

            # Category filter
            if cat_filter and entry.category != cat_filter:
                return 0.0, ""

            # Action+joint match
            target_action = f"{action}_{joint}"
            if target_action in entry.joint_actions:
                score += 10.0
                reasons.append(f"action:{target_action}")
            elif action and any(action in a for a in entry.joint_actions):
                score += 5.0
                reasons.append(f"partial_action:{action}")
            elif joint and any(joint in a for a in entry.joint_actions):
                score += 3.0
                reasons.append(f"joint:{joint}")

            if score == 0:
                return 0.0, ""

        # Token matching
        entry_tokens = set(entry.keywords)
        for token in tokens:
            if token in entry_tokens:
                score += 2.0
                reasons.append(f"keyword:{token}")
            elif any(token in ek for ek in entry_tokens):
                score += 1.0
                reasons.append(f"partial:{token}")

        # Category match bonus
        for token in tokens:
            cat = _CATEGORY_SYNONYMS.get(token)
            if cat and cat == entry.category:
                score += 1.5
                reasons.append(f"category:{cat}")

        # Region match bonus
        for token in tokens:
            region = _REGION_SYNONYMS.get(token)
            if region and region == entry.region:
                score += 1.0
                reasons.append(f"region:{region}")

        # Exact name match (highest bonus)
        query_joined = " ".join(tokens)
        if entry.display_name.lower() == query_joined:
            score += 20.0
            reasons.append("exact_name")

        return score, "; ".join(reasons)

    def _parse_functional_query(self, query: str
                                ) -> Optional[tuple[str, str, str]]:
        """Parse queries like 'muscles that flex the elbow'.

        Returns (category_filter, action, joint) or None.
        """
        # Pattern: [category] that [verb] the [joint]
        pattern = r"(\w+)\s+(?:that|which)\s+(\w+)\s+(?:the\s+)?(\w+)"
        m = re.search(pattern, query)
        if m:
            cat_word, verb, joint_word = m.groups()
            cat = _CATEGORY_SYNONYMS.get(cat_word, "")
            action = _ACTION_VERBS.get(verb, "")
            joint = _JOINT_SYNONYMS.get(joint_word, joint_word)
            if action and joint:
                return (cat, action, joint)

        return None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase tokens."""
        # Remove punctuation, split on spaces
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        return [t for t in cleaned.split() if len(t) > 1]

    @staticmethod
    def _infer_region(name: str, cfg: dict) -> str:
        """Infer anatomical region from muscle name and config."""
        name_lower = name.lower()
        if any(w in name_lower for w in ("biceps", "triceps", "brachialis",
                                         "deltoid", "forearm", "wrist")):
            return "arm"
        if any(w in name_lower for w in ("gluteus", "piriformis", "obturator")):
            return "pelvis"
        if any(w in name_lower for w in ("quadriceps", "hamstring", "gastrocnemius",
                                         "soleus", "tibialis", "peroneus",
                                         "vastus", "femor")):
            return "leg"
        if any(w in name_lower for w in ("trapezius", "rhomboid", "latissimus",
                                         "erector", "multifidus")):
            return "thorax"
        if any(w in name_lower for w in ("pectoralis", "pect.", "intercostal",
                                         "diaphragm", "oblique", "rectus abd",
                                         "transvers")):
            return "thorax"
        if any(w in name_lower for w in ("scm", "sternocleidomastoid", "scalene",
                                         "hyoid", "levator scap")):
            return "neck"
        return "thorax"

    @staticmethod
    def _infer_region_from_name(name: str) -> str:
        """Infer region from structure name."""
        name_lower = name.lower()
        if any(w in name_lower for w in ("skull", "cranium", "mandible",
                                         "maxilla", "orbit", "nasal")):
            return "head"
        if any(w in name_lower for w in ("cervical", "hyoid", "thyroid cart")):
            return "neck"
        if any(w in name_lower for w in ("thoracic", "rib", "sternum")):
            return "thorax"
        if any(w in name_lower for w in ("lumbar", "sacrum", "coccyx")):
            return "abdomen"
        if any(w in name_lower for w in ("ilium", "ischium", "pubis", "pelvis")):
            return "pelvis"
        if any(w in name_lower for w in ("humerus", "radius", "ulna",
                                         "scapula", "clavicle")):
            return "arm"
        if any(w in name_lower for w in ("femur", "tibia", "fibula", "patella")):
            return "leg"
        if any(w in name_lower for w in ("heart", "lung", "liver", "kidney",
                                         "stomach", "intestin", "spleen",
                                         "pancreas", "bladder")):
            return "abdomen"
        if any(w in name_lower for w in ("brain", "cerebr", "cerebel")):
            return "brain"
        if any(w in name_lower for w in ("aorta", "vena", "artery",
                                         "carotid", "jugular")):
            return "thorax"
        return ""

    @staticmethod
    def _infer_actions(cfg: dict) -> list[str]:
        """Infer joint actions from origin/insertion bones."""
        actions = []
        origin = cfg.get("originBones", [])
        insertion = cfg.get("insertionBones", [])
        all_bones = [b.lower() if isinstance(b, str) else ""
                     for b in (origin + insertion)]

        bone_text = " ".join(all_bones)

        # Shoulder muscles
        if any(b in bone_text for b in ("scapula", "humerus", "clavicle")):
            actions.extend(["flex_shoulder", "extend_shoulder",
                          "abduct_shoulder", "rotate_shoulder"])
        # Elbow muscles
        if any(b in bone_text for b in ("humerus", "radius", "ulna")):
            if "humerus" in bone_text and ("radius" in bone_text or "ulna" in bone_text):
                actions.extend(["flex_elbow", "extend_elbow"])
        # Wrist
        if any(b in bone_text for b in ("radius", "ulna")) and "carpal" in bone_text:
            actions.extend(["flex_wrist", "extend_wrist"])
        # Hip
        if any(b in bone_text for b in ("ilium", "ischium", "pubis", "pelvis")) and "femur" in bone_text:
            actions.extend(["flex_hip", "extend_hip",
                          "abduct_hip", "rotate_hip"])
        # Knee
        if "femur" in bone_text and ("tibia" in bone_text or "fibula" in bone_text):
            actions.extend(["flex_knee", "extend_knee"])
        # Ankle
        if ("tibia" in bone_text or "fibula" in bone_text) and ("talus" in bone_text or "calcaneus" in bone_text):
            actions.extend(["flex_ankle", "extend_ankle",
                          "invert_ankle", "evert_ankle"])
        # Spine
        if any(b in bone_text for b in ("vertebr", "sacrum", "thorac", "lumbar")):
            actions.extend(["flex_spine", "extend_spine", "rotate_spine"])

        return list(set(actions))
