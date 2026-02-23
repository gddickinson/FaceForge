"""Custom Animation Timeline Editor dialog.

Provides a visual keyframe editor for creating and editing AnimationClips
with per-DOF tracks, drag-and-drop keyframes, and preview playback.
"""

import json
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QComboBox, QLabel, QTreeWidget, QTreeWidgetItem,
    QFileDialog, QDoubleSpinBox, QToolBar, QScrollArea, QWidget,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QAction

from faceforge.scene.scene_animation import (
    AnimationClip, AnimationKeyframe, AnimationPlayer,
    clip_to_dict, load_clip_from_dict,
)
from faceforge.ui.widgets.timeline_canvas import TimelineCanvas

logger = logging.getLogger(__name__)


class TimelineEditor(QDialog):
    """Timeline editor dialog for creating/editing animation clips.

    Signals
    -------
    clip_changed(AnimationClip)
        Emitted when the clip is modified.
    preview_requested(AnimationClip)
        Emitted to preview the clip in the main viewport.
    """

    clip_changed = Signal(object)
    preview_requested = Signal(object)

    def __init__(self, parent=None, state=None):
        super().__init__(parent)
        self.setWindowTitle("Animation Timeline Editor")
        self.setMinimumSize(800, 500)
        self.resize(900, 550)

        self._state = state
        self._clip = AnimationClip(name="untitled")
        self._preview_player = AnimationPlayer()
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(16)
        self._preview_timer.timeout.connect(self._on_preview_tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ──
        toolbar = QToolBar()
        toolbar.setMovable(False)

        # Clip name
        toolbar.addWidget(QLabel("Clip:"))
        self._clip_name = QComboBox()
        self._clip_name.setEditable(True)
        self._clip_name.setMinimumWidth(120)
        self._clip_name.addItem("untitled")
        toolbar.addWidget(self._clip_name)

        toolbar.addSeparator()

        # File operations
        new_action = QAction("New", self)
        new_action.triggered.connect(self._new_clip)
        toolbar.addAction(new_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self._save_clip)
        toolbar.addAction(save_action)

        load_action = QAction("Load", self)
        load_action.triggered.connect(self._load_clip)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        # Capture pose
        capture_action = QAction("Capture Pose", self)
        capture_action.triggered.connect(self._capture_pose)
        toolbar.addAction(capture_action)

        # Delete keyframe
        delete_action = QAction("Delete Keyframe", self)
        delete_action.triggered.connect(self._delete_selected_keyframe)
        toolbar.addAction(delete_action)

        toolbar.addSeparator()

        # Duration
        toolbar.addWidget(QLabel("Duration:"))
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(1.0, 300.0)
        self._duration_spin.setValue(20.0)
        self._duration_spin.setSuffix("s")
        self._duration_spin.setDecimals(1)
        toolbar.addWidget(self._duration_spin)

        layout.addWidget(toolbar)

        # ── Transport controls ──
        transport_row = QHBoxLayout()

        self._play_btn = QPushButton("Play")
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._toggle_preview)
        transport_row.addWidget(self._play_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop_preview)
        transport_row.addWidget(self._stop_btn)

        self._time_label = QLabel("00:00.0 / 00:20.0")
        self._time_label.setFont(QFont("Courier New", 10))
        transport_row.addWidget(self._time_label)

        transport_row.addStretch()

        # Speed
        transport_row.addWidget(QLabel("Speed:"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 4.0)
        self._speed_spin.setValue(1.0)
        self._speed_spin.setSingleStep(0.1)
        transport_row.addWidget(self._speed_spin)

        layout.addLayout(transport_row)

        # ── Main content: track list + timeline canvas ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Track list (tree)
        self._track_tree = QTreeWidget()
        self._track_tree.setHeaderLabels(["DOF Track"])
        self._track_tree.setMaximumWidth(200)
        self._track_tree.setMinimumWidth(150)
        self._build_track_tree()
        splitter.addWidget(self._track_tree)

        # Right: Timeline canvas
        self._canvas = TimelineCanvas(self._clip)
        self._canvas.keyframe_moved.connect(self._on_keyframe_moved)
        self._canvas.keyframe_selected.connect(self._on_keyframe_selected)

        canvas_scroll = QScrollArea()
        canvas_scroll.setWidgetResizable(True)
        canvas_scroll.setWidget(self._canvas)
        splitter.addWidget(canvas_scroll)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # ── Keyframe properties ──
        props_row = QHBoxLayout()
        props_row.addWidget(QLabel("Time:"))
        self._kf_time_spin = QDoubleSpinBox()
        self._kf_time_spin.setRange(0.0, 300.0)
        self._kf_time_spin.setDecimals(2)
        self._kf_time_spin.valueChanged.connect(self._on_kf_time_changed)
        props_row.addWidget(self._kf_time_spin)

        props_row.addWidget(QLabel("Easing:"))
        self._kf_easing = QComboBox()
        self._kf_easing.addItems(["ease_in_out", "linear", "ease_in", "ease_out"])
        self._kf_easing.currentTextChanged.connect(self._on_kf_easing_changed)
        props_row.addWidget(self._kf_easing)

        props_row.addStretch()
        layout.addLayout(props_row)

    def _build_track_tree(self) -> None:
        """Build the DOF track tree."""
        tree = self._track_tree
        tree.clear()

        spine = QTreeWidgetItem(tree, ["Spine"])
        for dof in ["Flex", "Lat Bend", "Rotation"]:
            QTreeWidgetItem(spine, [dof])
        spine.setExpanded(False)

        for side in ["R", "L"]:
            shoulder = QTreeWidgetItem(tree, [f"Shoulder {side}"])
            for dof in ["Abduct", "Flex", "Rotate"]:
                QTreeWidgetItem(shoulder, [dof])

        for side in ["R", "L"]:
            elbow = QTreeWidgetItem(tree, [f"Elbow {side}"])
            QTreeWidgetItem(elbow, ["Flex"])

        for side in ["R", "L"]:
            hip = QTreeWidgetItem(tree, [f"Hip {side}"])
            for dof in ["Flex", "Abduct", "Rotate"]:
                QTreeWidgetItem(hip, [dof])

        for side in ["R", "L"]:
            knee = QTreeWidgetItem(tree, [f"Knee {side}"])
            QTreeWidgetItem(knee, ["Flex"])

        face_aus = QTreeWidgetItem(tree, ["Face AUs"])
        face_aus.setExpanded(False)
        head = QTreeWidgetItem(tree, ["Head"])
        for dof in ["Yaw", "Pitch", "Roll"]:
            QTreeWidgetItem(head, [dof])

        wrapper = QTreeWidgetItem(tree, ["Wrapper Pos"])

    def _new_clip(self) -> None:
        self._clip = AnimationClip(name="untitled")
        self._clip_name.setCurrentText("untitled")
        self._canvas.set_clip(self._clip)
        self._update_time_label(0.0)

    def _save_clip(self) -> None:
        self._clip.name = self._clip_name.currentText()
        data = clip_to_dict(self._clip)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation Clip",
            f"{self._clip.name}.json",
            "JSON (*.json)",
        )
        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved clip to %s", path)

    def _load_clip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Animation Clip", "", "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._clip = load_clip_from_dict(data)
            self._clip_name.setCurrentText(self._clip.name)
            self._duration_spin.setValue(self._clip.duration or 20.0)
            self._canvas.set_clip(self._clip)
            logger.info("Loaded clip: %s (%d keyframes)",
                         self._clip.name, len(self._clip.keyframes))
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _capture_pose(self) -> None:
        """Capture current body/face state as a keyframe."""
        if self._state is None:
            return

        time = self._canvas.cursor_time
        body = self._state.body
        face = self._state.face

        # Build body state dict
        body_dict = {}
        for field in (
            "spine_flex", "spine_lat_bend", "spine_rotation",
            "shoulder_r_abduct", "shoulder_r_flex", "shoulder_r_rotate",
            "shoulder_l_abduct", "shoulder_l_flex", "shoulder_l_rotate",
            "elbow_r_flex", "elbow_l_flex",
            "hip_r_flex", "hip_r_abduct", "hip_r_rotate",
            "hip_l_flex", "hip_l_abduct", "hip_l_rotate",
            "knee_r_flex", "knee_l_flex",
            "ankle_r_flex", "ankle_l_flex",
        ):
            val = getattr(body, field, 0.0)
            if val != 0.0:
                body_dict[field] = val

        # Build face AU dict
        au_dict = {}
        for au_id in ("AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9",
                       "AU10", "AU12", "AU14", "AU15", "AU17", "AU20",
                       "AU22", "AU23", "AU24", "AU25", "AU26"):
            val = face.get_au(au_id)
            if val > 0.01:
                au_dict[au_id] = val

        head_dict = {}
        if face.head_yaw: head_dict["headYaw"] = face.head_yaw
        if face.head_pitch: head_dict["headPitch"] = face.head_pitch
        if face.head_roll: head_dict["headRoll"] = face.head_roll

        kf = AnimationKeyframe(
            time=time,
            body_state=body_dict if body_dict else None,
            face_aus=au_dict if au_dict else None,
            head_rotation=head_dict if head_dict else None,
        )

        # Insert keyframe at correct position
        self._clip.keyframes.append(kf)
        self._clip.keyframes.sort(key=lambda k: k.time)
        self._canvas.set_clip(self._clip)
        self.clip_changed.emit(self._clip)

    def _delete_selected_keyframe(self) -> None:
        idx = self._canvas.selected_keyframe_index
        if 0 <= idx < len(self._clip.keyframes):
            self._clip.keyframes.pop(idx)
            self._canvas.set_clip(self._clip)
            self.clip_changed.emit(self._clip)

    def _toggle_preview(self, checked: bool) -> None:
        if checked:
            self._preview_player.load(self._clip)
            self._preview_player.set_speed(self._speed_spin.value())
            self._preview_player.play()
            self._preview_timer.start()
            self._play_btn.setText("Pause")
            self.preview_requested.emit(self._clip)
        else:
            self._preview_player.pause()
            self._preview_timer.stop()
            self._play_btn.setText("Play")

    def _stop_preview(self) -> None:
        self._preview_player.stop()
        self._preview_timer.stop()
        self._play_btn.setChecked(False)
        self._play_btn.setText("Play")
        self._canvas.set_cursor_time(0.0)
        self._update_time_label(0.0)

    def _on_preview_tick(self) -> None:
        self._preview_player.tick(0.016)
        t = self._preview_player.current_time
        self._canvas.set_cursor_time(t)
        self._update_time_label(t)

        if not self._preview_player.is_playing:
            self._stop_preview()

    def _update_time_label(self, t: float) -> None:
        d = self._clip.duration or self._duration_spin.value()
        mins_t, secs_t = divmod(t, 60)
        mins_d, secs_d = divmod(d, 60)
        self._time_label.setText(
            f"{int(mins_t):02d}:{secs_t:04.1f} / {int(mins_d):02d}:{secs_d:04.1f}"
        )

    def _on_keyframe_moved(self, index: int, new_time: float) -> None:
        if 0 <= index < len(self._clip.keyframes):
            self._clip.keyframes[index].time = max(0.0, new_time)
            self._clip.keyframes.sort(key=lambda k: k.time)
            self._canvas.set_clip(self._clip)
            self.clip_changed.emit(self._clip)

    def _on_keyframe_selected(self, index: int) -> None:
        if 0 <= index < len(self._clip.keyframes):
            kf = self._clip.keyframes[index]
            self._kf_time_spin.blockSignals(True)
            self._kf_time_spin.setValue(kf.time)
            self._kf_time_spin.blockSignals(False)
            self._kf_easing.blockSignals(True)
            idx = self._kf_easing.findText(kf.easing)
            if idx >= 0:
                self._kf_easing.setCurrentIndex(idx)
            self._kf_easing.blockSignals(False)

    def _on_kf_time_changed(self, value: float) -> None:
        idx = self._canvas.selected_keyframe_index
        if 0 <= idx < len(self._clip.keyframes):
            self._clip.keyframes[idx].time = value
            self._clip.keyframes.sort(key=lambda k: k.time)
            self._canvas.set_clip(self._clip)

    def _on_kf_easing_changed(self, text: str) -> None:
        idx = self._canvas.selected_keyframe_index
        if 0 <= idx < len(self._clip.keyframes):
            self._clip.keyframes[idx].easing = text

    def set_animation_player(self, player: AnimationPlayer) -> None:
        """Set external animation player for preview playback."""
        self._preview_player = player

    def set_state_refs(self, state, camera=None) -> None:
        """Set state and camera references for pose capture."""
        self._state = state
        self._camera = camera
