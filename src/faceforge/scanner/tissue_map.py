"""Tissue classification and density/intensity tables for scan modes.

Maps mesh names and material colors to tissue types, then provides
mode-specific intensity values and colormaps for CT, MRI, X-ray,
and anatomical imaging.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Scan modes
MODES = ("ct", "mri_t1", "mri_t2", "xray", "anatomical")

# ── Tissue keywords ─────────────────────────────────────────────────
# Each list maps mesh-name substrings → tissue type.
_TISSUE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("bone", [
        "vertebra", "femur", "tibia", "fibula", "humerus", "radius", "ulna",
        "scapula", "clavicle", "pelvis", "sacrum", "coccyx", "rib", "sternum",
        "skull", "mandible", "maxilla", "hyoid", "patella", "calcaneus",
        "talus", "navicular", "cuboid", "cuneiform", "metatarsal", "metacarpal",
        "phalanx", "phalang", "carpal", "tarsal", "disc", "bone",
    ]),
    ("cartilage", [
        "cartilage", "meniscus", "labrum", "thyroid_cart", "nasal_cart",
    ]),
    ("muscle", [
        "muscle", "bicep", "tricep", "deltoid", "pector", "trapezius",
        "latissimus", "gluteus", "quadricep", "hamstring", "gastrocnemius",
        "soleus", "masseter", "temporalis", "pterygoid", "sternocleid",
        "oblique", "rectus", "transvers", "diaphragm", "scalene",
        "platysma", "orbicularis", "zygomaticus", "buccinator",
        "frontalis", "corrugator", "levator", "depressor", "mentalis",
        "risorius", "nasalis", "procerus", "mylohyoid", "geniohyoid",
        "stylohyoid", "omohyoid", "sternohyoid", "thyrohyoid",
        "longus", "suboccipital", "serratus", "rhomboid", "infraspinatus",
        "supraspinatus", "teres", "subscapularis", "psoas", "iliacus",
        "piriformis", "sartorius", "gracilis", "adductor", "abductor",
        "popliteus", "peroneus", "tibialis", "extensor", "flexor",
        "inteross", "lumbrical", "opponens", "supinator", "pronator",
        "brachialis", "brachioradialis", "anconeus", "coracobrachialis",
    ]),
    ("organ", [
        "heart", "lung", "liver", "kidney", "spleen", "stomach",
        "intestin", "colon", "pancreas", "bladder", "uterus", "prostate",
        "thyroid", "adrenal", "gallbladder", "esophag", "trache", "bronch",
        "larynx", "pharynx", "appendix", "cecum", "duodenum", "jejunum",
        "ileum", "sigmoid", "rectum",
    ]),
    ("brain", [
        "brain", "cerebr", "cerebel", "cortex", "hippocampus", "thalamus",
        "hypothalamus", "amygdala", "putamen", "caudate", "pallidum",
        "substantia", "pons", "medulla", "midbrain", "fornix", "corpus_callosum",
        "ventricle",
    ]),
    ("vessel", [
        "artery", "arter", "vein", "aorta", "vena_cava", "carotid",
        "jugular", "subclavian", "brachial", "radial_art", "ulnar_art",
        "femoral", "popliteal", "saphenous", "iliac", "renal_art",
        "hepatic", "pulmonary", "coronary", "vertebral_art", "basilar",
        "vascular", "vasc",
    ]),
    ("nerve", [
        "nerve", "plexus", "ganglion", "spinal_cord",
    ]),
    ("fat", [
        "fat", "adipose",
    ]),
    ("skin", [
        "skin", "dermis", "epidermis", "face_mesh",
    ]),
    ("fluid", [
        "fluid", "csf", "synovial",
    ]),
    ("ligament", [
        "ligament", "tendon", "fascia", "aponeurosis", "retinaculum",
    ]),
    ("eye", [
        "eyeball", "sclera", "cornea", "lens", "retina",
    ]),
    ("ear", [
        "ear", "pinna", "auricle",
    ]),
]

# ── Density / intensity tables (0-1 per tissue per mode) ────────────
_CT_TABLE = {
    "bone": 0.95, "cartilage": 0.60, "muscle": 0.45, "organ": 0.40,
    "brain": 0.38, "vessel": 0.42, "nerve": 0.35, "fat": 0.15,
    "skin": 0.35, "fluid": 0.10, "ligament": 0.50, "eye": 0.20,
    "ear": 0.55, "air": 0.0, "unknown": 0.30,
}

_MRI_T1_TABLE = {
    "bone": 0.10, "cartilage": 0.45, "muscle": 0.70, "organ": 0.60,
    "brain": 0.65, "vessel": 0.25, "nerve": 0.55, "fat": 0.90,
    "skin": 0.50, "fluid": 0.15, "ligament": 0.40, "eye": 0.45,
    "ear": 0.50, "air": 0.0, "unknown": 0.40,
}

_MRI_T2_TABLE = {
    "bone": 0.05, "cartilage": 0.50, "muscle": 0.30, "organ": 0.55,
    "brain": 0.50, "vessel": 0.60, "nerve": 0.40, "fat": 0.45,
    "skin": 0.35, "fluid": 0.95, "ligament": 0.30, "eye": 0.70,
    "ear": 0.35, "air": 0.0, "unknown": 0.35,
}

_XRAY_TABLE = {
    "bone": 0.90, "cartilage": 0.35, "muscle": 0.20, "organ": 0.15,
    "brain": 0.15, "vessel": 0.12, "nerve": 0.10, "fat": 0.08,
    "skin": 0.10, "fluid": 0.05, "ligament": 0.25, "eye": 0.10,
    "ear": 0.30, "air": 0.0, "unknown": 0.15,
}

_MODE_TABLES = {
    "ct": _CT_TABLE,
    "mri_t1": _MRI_T1_TABLE,
    "mri_t2": _MRI_T2_TABLE,
    "xray": _XRAY_TABLE,
}


class TissueMapper:
    """Classifies meshes by tissue type and provides mode-specific intensities."""

    @staticmethod
    def classify(mesh_name: str, material_color: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> str:
        """Classify a mesh into a tissue type by name keywords.

        Falls back to material color brightness if no keyword matches.
        """
        name_lower = mesh_name.lower().replace(" ", "_").replace("-", "_")

        for tissue, keywords in _TISSUE_KEYWORDS:
            for kw in keywords:
                if kw in name_lower:
                    return tissue

        # Fallback: brightness heuristic
        brightness = 0.299 * material_color[0] + 0.587 * material_color[1] + 0.114 * material_color[2]
        if brightness > 0.85:
            return "bone"
        if brightness > 0.6:
            return "cartilage"
        if brightness < 0.2:
            return "vessel"
        return "muscle"

    @staticmethod
    def get_value(tissue: str, mode: str) -> float:
        """Return 0-1 intensity for a tissue type in a given scan mode."""
        table = _MODE_TABLES.get(mode)
        if table is None:
            return 0.5  # anatomical mode doesn't use this
        return table.get(tissue, table.get("unknown", 0.3))

    @staticmethod
    def get_colormap(mode: str):
        """Return a function mapping float (0-1) to RGB tuple (0-255 per channel).

        CT/X-ray: grayscale, MRI: grayscale, Anatomical: identity (unused).
        """
        if mode in ("ct", "xray"):
            return _grayscale_colormap
        if mode in ("mri_t1", "mri_t2"):
            return _mri_colormap
        # anatomical — identity
        return _anatomical_colormap


def _grayscale_colormap(value: float) -> tuple[int, int, int]:
    """Standard grayscale: 0=black, 1=white."""
    v = int(np.clip(value * 255, 0, 255))
    return (v, v, v)


def _mri_colormap(value: float) -> tuple[int, int, int]:
    """MRI-style grayscale with slight warm tint for mid-values."""
    v = np.clip(value, 0.0, 1.0)
    r = int(np.clip(v * 255 + v * (1 - v) * 20, 0, 255))
    g = int(np.clip(v * 255, 0, 255))
    b = int(np.clip(v * 245, 0, 255))
    return (r, g, b)


def _anatomical_colormap(value: float) -> tuple[int, int, int]:
    """Placeholder — anatomical mode uses mesh colors directly."""
    v = int(np.clip(value * 255, 0, 255))
    return (v, v, v)
