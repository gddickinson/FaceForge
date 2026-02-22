"""Material definitions for rendering."""

from enum import Enum, auto
from dataclasses import dataclass, field


class RenderMode(Enum):
    # Standard clinical
    SOLID = auto()
    WIREFRAME = auto()
    XRAY = auto()
    POINTS = auto()
    OPAQUE = auto()
    # Textbook / illustration
    ILLUSTRATION = auto()
    SEPIA = auto()
    COLOR_ATLAS = auto()
    PEN_INK = auto()
    MEDICAL = auto()
    # Creative / stylised
    HOLOGRAM = auto()
    CARTOON = auto()
    PORCELAIN = auto()
    BLUEPRINT = auto()
    THERMAL = auto()
    ETHEREAL = auto()


@dataclass
class Material:
    """Rendering material properties."""
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    opacity: float = 1.0
    shininess: float = 30.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: RenderMode = RenderMode.SOLID
    double_sided: bool = False
    transparent: bool = False
    visible: bool = True
    depth_write: bool = True
    wireframe_color: tuple[float, float, float] | None = None
    vertex_colors_active: bool = False

    @staticmethod
    def from_hex(color_int: int, **kwargs) -> "Material":
        """Create material from integer hex color (e.g., 0xd4a574)."""
        r = ((color_int >> 16) & 0xFF) / 255.0
        g = ((color_int >> 8) & 0xFF) / 255.0
        b = (color_int & 0xFF) / 255.0
        return Material(color=(r, g, b), **kwargs)

    @staticmethod
    def hex_to_rgb(color_int: int) -> tuple[float, float, float]:
        r = ((color_int >> 16) & 0xFF) / 255.0
        g = ((color_int >> 8) & 0xFF) / 255.0
        b = (color_int & 0xFF) / 255.0
        return (r, g, b)
