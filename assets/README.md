# assets/ -- Project Assets

Static data files used by FaceForge: JSON configuration, embedded mesh data, and BodyParts3D STL files.

## Directory Structure

```
assets/
  config/                  # JSON configuration files
    muscles/               # Per-region muscle definitions
    skeleton/              # Per-region skeleton definitions
  meshdata/                # Embedded mesh data (JSON format)
  stl -> ../../bodyparts3D/stl  # Symlink to BodyParts3D STL repository
```

## config/

JSON configuration files loaded by `core.config_loader`.

### Top-Level Configs

| File | Purpose |
|------|---------|
| `au_definitions.json` | Action Unit metadata (12 AUs) |
| `body_joint_limits.json` | Physiological joint limit ranges for body DOFs |
| `body_poses.json` | 6 body pose presets (anatomical, relaxed, walking, sitting, reaching, crouching) |
| `brain.json` | 80 brain structure STL definitions |
| `coordinate_transform.json` | BP3D-to-skull coordinate mapping (center, scale, skull center) |
| `expressions.json` | 12 expression presets with AU and head rotation values |
| `eye_colors.json` | Eye color preset definitions |
| `face_features.json` | 8 face feature STL definitions (eyes, ears, nose, eyebrows, throat) |
| `face_regions.json` | Vertex index lists per facial region for FACS displacement |
| `joint_limits.json` | Neck joint limit ranges |
| `ligaments.json` | Ligament STL definitions |
| `organs.json` | 52 organ structure STL definitions |
| `skin.json` | Skin mesh configuration |
| `skull_bones.json` | Individual skull bone STL definitions for BP3D mode |
| `teeth.json` | Individual tooth STL definitions |
| `vascular.json` | 50 vascular structure STL definitions |

### config/muscles/

Per-region muscle STL definitions. Each file contains an array of dicts with `stl` (filename), `color` (hex int), `auMap` or `lowerAttach` fields.

| File | Muscle Count | Region |
|------|-------------|--------|
| `jaw_muscles.json` | 22 | Masseters, temporalis, pterygoids, suprahyoids |
| `expression_muscles.json` | 38 | Facial expression muscles + 4 scalp muscles |
| `neck_muscles.json` | 36 | SCM, infrahyoid, scalenes, levator scapulae, deep prevertebral, suboccipital |
| `back_muscles.json` | 56 | Back muscles |
| `shoulder_muscles.json` | 20 | Shoulder muscles |
| `arm_muscles.json` | 62 | Arm and forearm muscles |
| `torso_muscles.json` | 31 | Torso muscles |
| `hip_muscles.json` | 24 | Hip muscles |
| `leg_muscles.json` | 60 | Leg muscles |
| `foot_muscles.json` | Varies | Foot muscles |
| `hand_muscles.json` | Varies | Hand muscles |

### config/skeleton/

Per-region skeleton STL definitions with pivot levels and fraction tables.

| File | Purpose |
|------|---------|
| `cervical_vertebrae.json` | C1-C7 + T1 + 7 discs (15 STLs) |
| `thoracic_spine.json` | T1-T12 vertebrae (22 STLs) |
| `thoracic_fractions.json` | Spine rotation distribution for T1-T12 |
| `lumbar_spine.json` | L1-L5 vertebrae (11 STLs) |
| `lumbar_fractions.json` | Spine rotation distribution for L1-L5 |
| `vertebra_fractions.json` | Cervical vertebra rotation fractions (C1-T1) |
| `rib_cage.json` | Ribs and sternum (43 STLs) |
| `pelvis.json` | Pelvis bones (2 STLs) |
| `upper_limb.json` | Humerus, radius, ulna, scapula, clavicle (10 STLs) |
| `lower_limb.json` | Femur, tibia, fibula, patella (8 STLs) |
| `hand.json` | Carpals, metacarpals, phalanges (54 STLs) |
| `foot.json` | Tarsals, metatarsals, phalanges (54 STLs) |

## meshdata/

Embedded mesh data in JSON format, extracted from the original single-file HTML application.

| File | Size | Contents |
|------|------|----------|
| `skull.json` | ~2.4 MB | Skull vertex positions + per-group triangle indices (cranium, jaw, upper teeth, lower teeth) |
| `mesh_data.json` | ~2.4 MB | Original embedded mesh data from the HTML version |
| `face.json` | ~21 KB | 468-vertex MediaPipe face mesh (positions + triangles) |
| `landmarks.json` | ~735 B | Face landmark reference points |
| `keyPoints.json` | ~115 B | Key anatomical point positions |

## stl/

Symlink to the BodyParts3D STL file repository (`../../bodyparts3D/stl`). Contains ~934 binary STL files representing individual anatomical structures identified by FMA (Foundational Model of Anatomy) codes.

File naming convention: `FMA{id}.stl` (e.g., `FMA49027.stl` for the right masseter muscle).

These files are not included in the repository and must be obtained separately from the BodyParts3D project (DBCLS, Japan).
