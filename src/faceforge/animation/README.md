# animation/ -- Animation Subsystems

This package provides autonomous animation behaviors and state management for smooth transitions. These systems run in the per-frame simulation loop and modify `FaceState` / `BodyState` values that are then consumed by the anatomy and body packages for mesh deformation.

All modules are pure Python with no GL dependencies.

## Modules

### `interpolation.py` -- State Interpolation

Smoothly interpolates face and body state values toward their targets using exponential decay (lerp per frame).

**Key class:** `StateInterpolator`

Interpolation speeds (units/second):
- AUs: 8.0 (fast for responsive expressions)
- Head rotation: 6.0
- Ear wiggle: 5.0
- Body joints: 4.0 (slower for natural body motion)
- Blink: 20.0 (very fast for snap blinks)

**Method:** `interpolate(face, target_au, target_head, target_ear_wiggle, body, target_body, dt)`

### `auto_blink.py` -- Automatic Blinking

Generates natural-looking blink patterns with random intervals.

**Key class:** `AutoBlink`
- Random interval between 2-6 seconds
- Close duration: 0.08s, Open duration: 0.12s
- Modifies `face.blink_amount` (0.0 to 1.0)
- Controlled by `face.auto_blink` toggle

### `auto_breathing.py` -- Automatic Breathing

Subtle breathing animation on the face using sinusoidal rhythm.

**Key class:** `AutoBreathing`
- Breathing rate: ~0.4 Hz (2.5 rad/s phase increment)
- Cycles AU9 (nostril flare, amplitude 0.06) and AU25 (lip part, amplitude 0.03)
- Uses `max()` to avoid overriding stronger expression values
- Controlled by `face.auto_breathing` toggle

### `eye_tracking.py` -- Eye Tracking

Updates eye look direction based on mouse/cursor position.

**Key class:** `EyeTracking`
- Maps normalized screen coordinates (-1..1) to `eyeLookX`/`eyeLookY` targets
- Sensitivity: 0.8, Smoothing: 6.0
- Controlled by `face.eye_tracking` toggle

### `micro_expressions.py` -- Micro-Expression Generator

Random subtle AU flickers for lifelike appearance.

**Key class:** `MicroExpressionGen`
- Fires every 1-4 seconds, selects a random AU
- Adds small intensity (0.02 to 0.15) for 0.2-0.5 seconds
- Restores original target value after duration
- Controlled by `face.micro_expressions` toggle

### `preset_manager.py` -- Expression and Pose Presets

Manages expression and body pose presets loaded from JSON config files.

**Key class:** `PresetManager`
- Loads expressions from `expressions.json` (12 presets)
- Loads body poses from `body_poses.json` (6 presets)
- `set_expression(name, state)` -- Resets all AU targets to 0, then applies preset values (including optional head rotation)
- `set_body_pose(name, state)` -- Applies body pose DOF values to target body state

## Integration

These systems are instantiated by the `Simulation` class (in `coordination/simulation.py`) and called in this order each frame:

1. `StateInterpolator.interpolate()` -- smooth toward targets
2. `AutoBlink.update()` -- generate blinks
3. `AutoBreathing.update()` -- subtle breathing
4. `MicroExpressionGen.update()` -- random AU flickers
5. `EyeTracking.update()` -- cursor-follow eyes

The `PresetManager` is called by UI event handlers when the user selects an expression or body pose.

## Internal Dependencies

- `core.state` -- FaceState, BodyState, TargetAU, TargetHead, StateManager, AU_IDS
- `core.config_loader` -- load_config for JSON preset files
