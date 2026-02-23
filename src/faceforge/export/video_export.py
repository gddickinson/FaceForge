"""Video/GIF export from the GL viewport.

Captures frames from the QOpenGLWidget and encodes to MP4 (via ffmpeg)
or GIF (via PIL/Pillow).
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class VideoExporter:
    """Captures frames from a GL widget and encodes to video.

    Parameters
    ----------
    gl_widget : GLViewport
        The OpenGL widget to grab frames from.
    """

    def __init__(self, gl_widget):
        self._gl_widget = gl_widget
        self._ffmpeg_available = _check_ffmpeg()

    @property
    def ffmpeg_available(self) -> bool:
        return self._ffmpeg_available

    def export_turntable(
        self,
        output_path: str,
        duration: float = 10.0,
        fps: int = 30,
        width: int = 0,
        height: int = 0,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Export a 360-degree turntable rotation.

        Parameters
        ----------
        output_path : str
            Output file path (.mp4 or .gif).
        duration : float
            Duration in seconds.
        fps : int
            Frames per second.
        width, height : int
            Output resolution (0 = current viewport size).
        on_progress : callable
            Progress callback (0.0 to 1.0).

        Returns
        -------
        bool
            True if export succeeded.
        """
        import math
        total_frames = int(duration * fps)
        if total_frames <= 0:
            return False

        camera = self._gl_widget.camera
        orbit = self._gl_widget.orbit_controls

        # Save original camera state
        orig_theta = orbit._theta

        frames = []
        for i in range(total_frames):
            # Rotate camera around target
            angle = (i / total_frames) * 2.0 * math.pi
            orbit._theta = orig_theta + angle
            orbit._update_camera()

            # Grab frame
            frame = self._grab_frame(width, height)
            if frame is not None:
                frames.append(frame)

            if on_progress:
                on_progress(i / total_frames)

        # Restore camera
        orbit._theta = orig_theta
        orbit._update_camera()

        if not frames:
            return False

        return self._encode(frames, output_path, fps, on_progress)

    def export_animation(
        self,
        anim_player,
        simulation,
        output_path: str,
        fps: int = 30,
        width: int = 0,
        height: int = 0,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Export current animation clip.

        Parameters
        ----------
        anim_player : AnimationPlayer
            The animation player with a loaded clip.
        simulation : Simulation
            The simulation to step.
        output_path : str
            Output file path.
        fps : int
            Frames per second.
        on_progress : callable
            Progress callback.
        """
        if anim_player is None or anim_player.duration <= 0:
            return False

        duration = anim_player.duration
        dt = 1.0 / fps
        total_frames = int(duration * fps)

        # Reset to beginning
        anim_player.stop()
        anim_player.play()

        frames = []
        for i in range(total_frames):
            # Step simulation
            simulation.step(dt)
            anim_player.tick(dt)

            # Render and grab
            self._gl_widget.paintGL()
            frame = self._grab_frame(width, height)
            if frame is not None:
                frames.append(frame)

            if on_progress:
                on_progress(i / total_frames)

        anim_player.stop()

        if not frames:
            return False

        return self._encode(frames, output_path, fps, on_progress)

    def export_screenshot(self, output_path: str,
                          width: int = 0, height: int = 0) -> bool:
        """Export a single frame as an image."""
        frame = self._grab_frame(width, height)
        if frame is None:
            return False

        try:
            frame.save(output_path)
            return True
        except Exception as e:
            logger.error("Screenshot failed: %s", e)
            return False

    def _grab_frame(self, width: int = 0, height: int = 0):
        """Grab the current frame from the GL widget.

        Returns a QImage or PIL Image, or None on failure.
        """
        try:
            qimage = self._gl_widget.grabFramebuffer()
            if qimage.isNull():
                return None

            if width > 0 and height > 0:
                from PySide6.QtCore import Qt
                qimage = qimage.scaled(width, height,
                                       Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
            return qimage
        except Exception as e:
            logger.warning("Frame grab failed: %s", e)
            return None

    def _encode(self, frames: list, output_path: str, fps: int,
                on_progress: Optional[Callable[[float], None]] = None) -> bool:
        """Encode frames to output file."""
        path = Path(output_path)
        suffix = path.suffix.lower()

        if suffix == ".gif":
            return self._encode_gif(frames, output_path, fps, on_progress)
        elif suffix in (".mp4", ".mkv", ".avi", ".mov"):
            if self._ffmpeg_available:
                return self._encode_ffmpeg(frames, output_path, fps, on_progress)
            else:
                # Fallback: save as PNG sequence
                return self._save_png_sequence(frames, path.parent / path.stem,
                                               on_progress)
        else:
            logger.error("Unsupported format: %s", suffix)
            return False

    def _encode_ffmpeg(self, frames: list, output_path: str, fps: int,
                       on_progress: Optional[Callable] = None) -> bool:
        """Encode frames to MP4 via ffmpeg pipe."""
        if not frames:
            return False

        first = frames[0]
        width = first.width()
        height = first.height()

        try:
            proc = subprocess.Popen(
                [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-pix_fmt", "rgba",
                    "-s", f"{width}x{height}",
                    "-r", str(fps),
                    "-i", "-",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "medium",
                    "-crf", "18",
                    output_path,
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            for i, frame in enumerate(frames):
                # Convert QImage to raw RGBA bytes
                frame = frame.convertToFormat(frame.Format.Format_RGBA8888)
                ptr = frame.constBits()
                raw = bytes(ptr)
                proc.stdin.write(raw)

                if on_progress:
                    on_progress(0.5 + 0.5 * (i / len(frames)))

            proc.stdin.close()
            proc.wait(timeout=60)

            if proc.returncode == 0:
                logger.info("Exported video: %s", output_path)
                return True
            else:
                stderr = proc.stderr.read().decode()
                logger.error("ffmpeg failed: %s", stderr[:500])
                return False

        except Exception as e:
            logger.error("ffmpeg encoding failed: %s", e)
            return False

    def _encode_gif(self, frames: list, output_path: str, fps: int,
                    on_progress: Optional[Callable] = None) -> bool:
        """Encode frames to GIF using PIL."""
        try:
            from PIL import Image
        except ImportError:
            logger.error("PIL/Pillow not available for GIF export")
            return False

        try:
            pil_frames = []
            for i, qimg in enumerate(frames):
                qimg = qimg.convertToFormat(qimg.Format.Format_RGBA8888)
                ptr = qimg.constBits()
                raw = bytes(ptr)
                img = Image.frombytes("RGBA", (qimg.width(), qimg.height()), raw)
                # Convert to P mode for GIF (with dithering)
                img = img.convert("RGB").quantize(colors=256, dither=1)
                pil_frames.append(img)

                if on_progress:
                    on_progress(0.5 + 0.5 * (i / len(frames)))

            if pil_frames:
                frame_duration = int(1000 / fps)
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=frame_duration,
                    loop=0,
                )
                logger.info("Exported GIF: %s", output_path)
                return True
            return False

        except Exception as e:
            logger.error("GIF encoding failed: %s", e)
            return False

    def _save_png_sequence(self, frames: list, output_dir: Path,
                           on_progress: Optional[Callable] = None) -> bool:
        """Save frames as a numbered PNG sequence (ffmpeg fallback)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            path = output_dir / f"frame_{i:04d}.png"
            frame.save(str(path))
            if on_progress:
                on_progress(0.5 + 0.5 * (i / len(frames)))
        logger.info("Saved %d frames to: %s", len(frames), output_dir)
        return True
