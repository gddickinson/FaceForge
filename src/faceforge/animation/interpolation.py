"""State interpolation for smooth transitions."""

from faceforge.core.state import FaceState, BodyState, TargetAU, TargetHead, AU_IDS


class StateInterpolator:
    """Smoothly interpolates face and body state toward targets.

    Uses exponential decay (lerp per frame) for natural-feeling transitions.
    """

    AU_SPEED = 8.0  # Interpolation speed for AUs
    HEAD_SPEED = 6.0
    EAR_SPEED = 5.0
    BODY_SPEED = 4.0
    BLINK_SPEED = 20.0  # Fast for blinks

    def interpolate(
        self,
        face: FaceState,
        target_au: TargetAU,
        target_head: TargetHead,
        target_ear_wiggle: float,
        body: BodyState,
        target_body: BodyState,
        dt: float,
    ) -> None:
        """Advance all state values toward their targets."""
        self._interpolate_aus(face, target_au, dt)
        self._interpolate_head(face, target_head, dt)
        self._interpolate_ear(face, target_ear_wiggle, dt)
        self._interpolate_body(body, target_body, dt)

    def _interpolate_aus(self, face: FaceState, target: TargetAU, dt: float) -> None:
        t = min(1.0, self.AU_SPEED * dt)
        for au_id in AU_IDS:
            current = face.get_au(au_id)
            goal = target.get(au_id)
            face.set_au(au_id, current + (goal - current) * t)

    def _interpolate_head(self, face: FaceState, target: TargetHead, dt: float) -> None:
        t = min(1.0, self.HEAD_SPEED * dt)
        face.head_yaw += (target.head_yaw - face.head_yaw) * t
        face.head_pitch += (target.head_pitch - face.head_pitch) * t
        face.head_roll += (target.head_roll - face.head_roll) * t

    def _interpolate_ear(self, face: FaceState, target: float, dt: float) -> None:
        t = min(1.0, self.EAR_SPEED * dt)
        face.ear_wiggle += (target - face.ear_wiggle) * t

    def _interpolate_body(self, body: BodyState, target: BodyState, dt: float) -> None:
        """Advance body state toward ``target``.

        Only the fields on ``BodyState.INTERPOLATED_FIELDS`` are lerped, and
        only ``BodyState.FLAG_FIELDS`` are copied.  This is an explicit
        allowlist rather than everything ``to_dict()`` happens to return: the
        old form dragged the ``breath_phase_body`` accumulator, the ``gender``
        slider and the six boolean ``auto_*`` toggles toward defaults that
        nothing ever assigns (see ``defects.md``).
        """
        t = min(1.0, self.BODY_SPEED * dt)
        for key in BodyState.INTERPOLATED_FIELDS:
            current = getattr(body, key)
            goal = getattr(target, key)
            setattr(body, key, current + (goal - current) * t)

        # Boolean toggles are state, not a trajectory: snap them so a path that
        # writes only target_body (preset_manager, animation clips, BODY_POSE_SET)
        # takes effect immediately and the field stays a real bool.
        for key in BodyState.FLAG_FIELDS:
            goal = bool(getattr(target, key))
            if getattr(body, key) is not goal:
                setattr(body, key, goal)
