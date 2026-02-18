# ui/ -- Qt GUI Framework

This package implements the desktop GUI using PySide6 (Qt 6). The interface is a dark-themed three-panel layout with a left info panel, central 3D viewport, and right tabbed control panel.

## Layout

```
+----------------------------------------------------------+
|  InfoPanel  |        GLViewport         |  ControlPanel  |
|  (AU bars)  |    (OpenGL 3D scene)      |  (6 tabs)      |
|             |                           |                 |
+----------------------------------------------------------+
|                     StatusBar                             |
+----------------------------------------------------------+
```

## Modules

### `main_window.py` -- Main Application Window

**Key class:** `MainWindow` (extends QMainWindow)
- Assembles InfoPanel, GLViewport, and ControlPanel in a horizontal layout
- Applies the DARK_THEME stylesheet
- Status bar shows vertex count, face count, and FPS
- Window title: "FaceForge -- Full-Body Anatomical Simulation"
- Default size: 1400x900

### `control_panel.py` -- Right Control Panel

**Key class:** `ControlPanel`
- Fixed width 330px
- Contains QTabWidget with 6 tabs: ANIMATE, BODY, LAYERS, ALIGN, DISPLAY, DEBUG
- Each tab is a separate widget from the `tabs/` subpackage

### `info_panel.py` -- Left Info Panel

**Key class:** `InfoPanel`
- Displays current expression name
- Shows all 12 AU values as labeled progress bars
- Updates from StateManager each frame

### `style.py` -- Dark Theme Stylesheet

QSS stylesheet matching the original HTML version's visual design.

Color palette:
- Background: `#0a0b0e`
- Surface: `#12141a` / `#1a1d26`
- Border: `#252830`
- Text: `#e8e9ed` / dim: `#8b8e99`
- Accent: `#4fd1c5` (teal) / `#f6ad55` (amber)

Exports `DARK_THEME` (str) and `COLORS` (dict).

## Tabs (`tabs/`)

### `animate_tab.py` -- Expression and Animation Controls

- Expression preset grid (12 expressions: neutral, happy, sad, angry, surprised, fear, disgust, contempt, pout, kiss, pain, thinking)
- AU sliders (12 Action Units, range 0-1)
- Head rotation sliders (yaw, pitch, roll)
- Ear wiggle slider
- Auto-animation toggles (blink, breathing, eye tracking, micro-expressions)
- Pupil dilation slider

### `body_tab.py` -- Body Pose and Joint Controls

- Body pose preset grid (6 poses: anatomical, relaxed, walking, sitting, reaching, crouching)
- Spine sliders: flex, side bend, twist
- Limb joint sliders: shoulder, elbow, wrist, hip, knee, ankle (bilateral)
- Breathing depth control
- Finger articulation controls

### `layers_tab.py` -- Layer Visibility Toggles

- Head layers: skull, jaw muscles, expression muscles, face features (eyes, ears, nose cartilage, eyebrows), neck muscles, vertebrae, throat
- Body layers: skeleton, body muscles, organs, vasculature, brain
- Skin mesh toggle
- Structure label toggle
- Each toggle emits `EventType.LAYER_TOGGLED`

### `align_tab.py` -- Face/Skull Alignment

- Scale, X/Y/Z offset, rotation sliders for face-skull alignment
- Reset to defaults button
- Real-time alignment updates via `EventType.ALIGNMENT_CHANGED`

### `display_tab.py` -- Rendering Options

- Render mode selector: Solid, Wireframe, X-Ray, Points, Opaque
- Camera preset buttons (body views, head views)
- Background color picker
- Mesh color picker
- Skull mode toggle (original vs BP3D)

### `debug_tab.py` -- Debug Tools

- Run skinning diagnostic button
- Diagnostic output display
- State inspection tools

## Widgets (`widgets/`)

Reusable UI components:

| Widget | Purpose |
|--------|---------|
| `slider_row.py` | Labeled slider with value display, configurable range and step |
| `toggle_row.py` | Labeled on/off toggle switch |
| `section_label.py` | Styled section header label |
| `collapsible_section.py` | Expandable/collapsible section with header and content |
| `expression_grid.py` | Grid of expression preset buttons |
| `eye_color_grid.py` | Grid of eye color preset buttons |
| `color_picker.py` | Color selection widget |
| `loading_overlay.py` | Translucent overlay showing loading progress with phase text |
| `label_overlay.py` | Floating text labels for anatomical structure names |

## External Dependencies

- `PySide6` -- QMainWindow, QWidget, QTabWidget, QSlider, QLabel, QProgressBar, QTimer, etc.

## Internal Dependencies

- `core.events` -- EventBus, EventType for publishing UI actions
- `core.state` -- StateManager, FaceState, BodyState for reading/writing state
- `rendering.gl_widget` -- GLViewport for the 3D viewport
