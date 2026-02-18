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
        t = min(1.0, self.BODY_SPEED * dt)
        body_dict = body.to_dict()
        target_dict = target.to_dict()
        for key in body_dict:
            if key in target_dict:
                current = body_dict[key]
                goal = target_dict[key]
                setattr(body, key, current + (goal - current) * t)
