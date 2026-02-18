#!/usr/bin/env python3
"""
Extract JavaScript data structures from faceforge-muscles.html into JSON config files.

Reads the monolithic HTML file and extracts embedded JS constants into
properly formatted JSON files under assets/config/ and assets/meshdata/.
"""

import json
import os
import re
import sys

HTML_PATH = os.path.join(os.path.dirname(__file__), '..', 'faceforge-muscles.html')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
CONFIG_DIR = os.path.join(ASSETS_DIR, 'config')
MESHDATA_DIR = os.path.join(ASSETS_DIR, 'meshdata')


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [
        CONFIG_DIR,
        os.path.join(CONFIG_DIR, 'muscles'),
        os.path.join(CONFIG_DIR, 'skeleton'),
        MESHDATA_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def read_html_lines():
    """Read the HTML file and return all lines."""
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        return f.readlines()


def js_to_json(js_text):
    """
    Convert a JavaScript object/array literal to valid JSON.
    Handles: unquoted keys, single-quoted strings, trailing commas,
    hex integers (0xRRGGBB), JS comments.
    """
    text = js_text

    # Remove single-line comments (// ...) but not inside strings
    # We do a careful approach: process line by line
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove // comments that aren't inside a string
        result = []
        in_single = False
        in_double = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '\\' and i + 1 < len(line) and (in_single or in_double):
                result.append(ch)
                result.append(line[i + 1])
                i += 2
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                result.append(ch)
            elif ch == '"' and not in_single:
                in_double = not in_double
                result.append(ch)
            elif ch == '/' and i + 1 < len(line) and line[i + 1] == '/' and not in_single and not in_double:
                break  # rest of line is comment
            else:
                result.append(ch)
            i += 1
        cleaned_lines.append(''.join(result))
    text = '\n'.join(cleaned_lines)

    # Convert hex integers: 0xRRGGBB -> decimal integer
    # Must handle both 0x and negative hex values
    def hex_to_int(m):
        return str(int(m.group(0), 16))
    text = re.sub(r'0x[0-9a-fA-F]+', hex_to_int, text)

    # Replace single quotes with double quotes
    # First, temporarily protect escaped single quotes inside strings
    text = text.replace("\\'", "@@ESCAPED_SQUOTE@@")
    text = text.replace("'", '"')
    text = text.replace("@@ESCAPED_SQUOTE@@", "'")

    # Add double quotes around unquoted keys
    # Match patterns like `{key:` or `,key:` or start-of-object keys
    # Unquoted key: word chars possibly with dots, followed by :
    text = re.sub(
        r'(?<=[{,\n])\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:',
        r' "\1":',
        text
    )

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Handle JS boolean/null (already valid in JSON, but just in case)
    # true, false, null are the same

    return text


def extract_block(lines, start_pattern, end_chars='];', start_line_hint=None):
    """
    Extract a JS block starting with `const VARNAME = ...` and ending with end_chars.
    Returns the content between = and end_chars (inclusive of brackets).

    start_line_hint: 1-based line number hint to start searching near
    """
    start_idx = 0
    if start_line_hint:
        start_idx = max(0, start_line_hint - 5)  # search a few lines before hint

    found_start = None
    for i in range(start_idx, len(lines)):
        if re.search(start_pattern, lines[i]):
            found_start = i
            break

    if found_start is None:
        raise ValueError(f"Could not find pattern: {start_pattern}")

    # Find the = sign on this line
    line = lines[found_start]
    eq_pos = line.index('=')
    # Get everything after the = sign
    content_start = line[eq_pos + 1:].strip()

    # Determine bracket type
    if content_start.startswith('['):
        open_bracket, close_bracket = '[', ']'
    elif content_start.startswith('{'):
        open_bracket, close_bracket = '{', '}'
    else:
        raise ValueError(f"Expected [ or {{ after = on line {found_start + 1}: {content_start[:50]}")

    # Accumulate content, tracking bracket depth
    depth = 0
    content_parts = []
    for i in range(found_start, len(lines)):
        if i == found_start:
            text = line[eq_pos + 1:]
        else:
            text = lines[i]

        for ch in text:
            if ch == open_bracket:
                depth += 1
            elif ch == close_bracket:
                depth -= 1

        content_parts.append(text if i != found_start else line[eq_pos + 1:])

        if depth == 0:
            break

    content = ''.join(content_parts).strip()
    # Remove trailing semicolon
    if content.endswith(';'):
        content = content[:-1].strip()

    return content


def extract_single_line_object(lines, pattern, start_hint=None):
    """Extract a simple single-line const like `const X = {x:1, y:2, z:3};`"""
    start_idx = 0
    if start_hint:
        start_idx = max(0, start_hint - 5)

    for i in range(start_idx, len(lines)):
        if re.search(pattern, lines[i]):
            line = lines[i]
            eq_pos = line.index('=')
            content = line[eq_pos + 1:].strip()
            # Remove trailing comment
            comment_pos = content.find('//')
            if comment_pos >= 0:
                content = content[:comment_pos].strip()
            if content.endswith(';'):
                content = content[:-1].strip()
            return content
    raise ValueError(f"Could not find: {pattern}")


def extract_mesh_data(lines):
    """
    Extract the MESH_DATA object from line 381 (0-indexed: 380).
    This is a huge single-line JS object.
    """
    for i, line in enumerate(lines):
        if 'const MESH_DATA' in line:
            eq_pos = line.index('=')
            content = line[eq_pos + 1:].strip()
            if content.endswith(';'):
                content = content[:-1].strip()
            return content
    raise ValueError("Could not find MESH_DATA")


def parse_and_write(js_content, output_path, description=""):
    """Convert JS content to JSON and write to file."""
    json_text = js_to_json(js_content)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"  ERROR parsing {description}: {e}")
        # Write the problematic text for debugging
        debug_path = output_path + '.debug'
        with open(debug_path, 'w') as f:
            f.write(json_text)
        print(f"  Debug text written to {debug_path}")
        return None

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    count = len(data) if isinstance(data, (list, dict)) else '?'
    print(f"  Wrote {output_path} ({count} entries)")
    return data


def main():
    print("Reading HTML file...")
    lines = read_html_lines()
    print(f"  {len(lines)} lines read from {HTML_PATH}")

    ensure_dirs()

    # ──────────────────────────────────────────────
    # 1. AU_INFO (line ~401)
    # ──────────────────────────────────────────────
    print("\nExtracting AU_INFO...")
    js = extract_block(lines, r'const AU_INFO\s*=', start_line_hint=401)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'au_definitions.json'), 'AU_INFO')

    # ──────────────────────────────────────────────
    # 2. EXPRESSIONS (line ~416)
    # ──────────────────────────────────────────────
    print("Extracting EXPRESSIONS...")
    js = extract_block(lines, r'const EXPRESSIONS\s*=', start_line_hint=416)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'expressions.json'), 'EXPRESSIONS')

    # ──────────────────────────────────────────────
    # 3. FACE_REGIONS (line ~432)
    # ──────────────────────────────────────────────
    print("Extracting FACE_REGIONS...")
    js = extract_block(lines, r'const FACE_REGIONS\s*=', start_line_hint=432)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'face_regions.json'), 'FACE_REGIONS')

    # ──────────────────────────────────────────────
    # 4. FACE_FEATURE_DEFS (line ~958)
    # ──────────────────────────────────────────────
    print("Extracting FACE_FEATURE_DEFS...")
    js = extract_block(lines, r'const FACE_FEATURE_DEFS\s*=', start_line_hint=958)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'face_features.json'), 'FACE_FEATURE_DEFS')

    # ──────────────────────────────────────────────
    # 5. JOINT_LIMITS + MUSCLE_LIMITS (lines ~478, ~492)
    # ──────────────────────────────────────────────
    print("Extracting JOINT_LIMITS and MUSCLE_LIMITS...")
    js_joint = extract_block(lines, r'const JOINT_LIMITS\s*=', start_line_hint=478)
    js_muscle = extract_block(lines, r'const MUSCLE_LIMITS\s*=', start_line_hint=492)
    joint_json = js_to_json(js_joint)
    muscle_json = js_to_json(js_muscle)
    combined = {
        "joint_limits": json.loads(joint_json),
        "muscle_limits": json.loads(muscle_json)
    }
    outpath = os.path.join(CONFIG_DIR, 'joint_limits.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)
    print(f"  Wrote {outpath}")

    # ──────────────────────────────────────────────
    # 6. Coordinate transform constants (lines ~826-828, ~854)
    # ──────────────────────────────────────────────
    print("Extracting coordinate transform constants...")
    bp3d_center = extract_single_line_object(lines, r'const BP3D_CENTER\s*=', start_hint=826)
    skull_center = extract_single_line_object(lines, r'const SKULL_CENTER\s*=', start_hint=827)
    bp3d_scale = extract_single_line_object(lines, r'const BP3D_SCALE\s*=', start_hint=828)
    stl_base_line = None
    for i, line in enumerate(lines):
        if "const STL_BASE" in line:
            stl_base_line = line
            break
    stl_base_val = 'bodyparts3D/stl/'
    if stl_base_line:
        m = re.search(r"['\"]([^'\"]+)['\"]", stl_base_line)
        if m:
            stl_base_val = m.group(1)

    coord_data = {
        "bp3d_center": json.loads(js_to_json(bp3d_center)),
        "skull_center": json.loads(js_to_json(skull_center)),
        "bp3d_scale": json.loads(js_to_json(bp3d_scale)),
        "stl_base": stl_base_val
    }
    outpath = os.path.join(CONFIG_DIR, 'coordinate_transform.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(coord_data, f, indent=2)
    print(f"  Wrote {outpath}")

    # ──────────────────────────────────────────────
    # 7. BODY_POSES (line ~4849)
    # ──────────────────────────────────────────────
    print("Extracting BODY_POSES...")
    js = extract_block(lines, r'const BODY_POSES\s*=', start_line_hint=4849)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'body_poses.json'), 'BODY_POSES')

    # ──────────────────────────────────────────────
    # 8. MUSCLE_STL_DEFS — jaw muscles (line ~855)
    # ──────────────────────────────────────────────
    print("Extracting MUSCLE_STL_DEFS (jaw muscles)...")
    js = extract_block(lines, r'const MUSCLE_STL_DEFS\s*=', start_line_hint=855)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'muscles', 'jaw_muscles.json'), 'MUSCLE_STL_DEFS')

    # ──────────────────────────────────────────────
    # 9. EXPR_MUSCLE_DEFS — expression muscles (line ~892)
    # ──────────────────────────────────────────────
    print("Extracting EXPR_MUSCLE_DEFS (expression muscles)...")
    js = extract_block(lines, r'const EXPR_MUSCLE_DEFS\s*=', start_line_hint=892)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'muscles', 'expression_muscles.json'), 'EXPR_MUSCLE_DEFS')

    # ──────────────────────────────────────────────
    # 10. NECK_MUSCLE_DEFS (line ~972)
    # ──────────────────────────────────────────────
    print("Extracting NECK_MUSCLE_DEFS...")
    js = extract_block(lines, r'const NECK_MUSCLE_DEFS\s*=', start_line_hint=972)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'muscles', 'neck_muscles.json'), 'NECK_MUSCLE_DEFS')

    # ──────────────────────────────────────────────
    # 11. Body muscle arrays
    # ──────────────────────────────────────────────
    body_muscles = [
        ('BACK_MUSCLE_DEFS', 1296, 'back_muscles.json'),
        ('SHOULDER_MUSCLE_DEFS', 1356, 'shoulder_muscles.json'),
        ('ARM_MUSCLE_DEFS', 1379, 'arm_muscles.json'),
        ('TORSO_MUSCLE_DEFS', 1444, 'torso_muscles.json'),
        ('HIP_MUSCLE_DEFS', 1478, 'hip_muscles.json'),
        ('LEG_MUSCLE_DEFS', 1505, 'leg_muscles.json'),
    ]
    for varname, hint, filename in body_muscles:
        print(f"Extracting {varname}...")
        js = extract_block(lines, rf'const {varname}\s*=', start_line_hint=hint)
        parse_and_write(js, os.path.join(CONFIG_DIR, 'muscles', filename), varname)

    # ──────────────────────────────────────────────
    # 12. Skeleton arrays
    # ──────────────────────────────────────────────
    skeleton_arrays = [
        ('VERTEBRA_DEFS', 1033, 'cervical_vertebrae.json'),
        ('VERTEBRA_FRACTIONS', 1052, 'vertebra_fractions.json'),
        ('THORACIC_DEFS', 1068, 'thoracic_spine.json'),
        ('LUMBAR_DEFS', 1093, 'lumbar_spine.json'),
        ('RIB_CAGE_DEFS', 1107, 'rib_cage.json'),
        ('PELVIS_DEFS', 1153, 'pelvis.json'),
        ('UPPER_LIMB_SKEL_DEFS', 1158, 'upper_limb.json'),
        ('HAND_SKEL_DEFS', 1171, 'hand.json'),
        ('LOWER_LIMB_SKEL_DEFS', 1228, 'lower_limb.json'),
        ('FOOT_SKEL_DEFS', 1239, 'foot.json'),
    ]
    for varname, hint, filename in skeleton_arrays:
        print(f"Extracting {varname}...")
        js = extract_block(lines, rf'const {varname}\s*=', start_line_hint=hint)
        parse_and_write(js, os.path.join(CONFIG_DIR, 'skeleton', filename), varname)

    # Thoracic/lumbar fractions (line ~3893, ~3896)
    print("Extracting THORACIC_FRACTIONS...")
    js = extract_block(lines, r'const THORACIC_FRACTIONS\s*=', start_line_hint=3893)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'skeleton', 'thoracic_fractions.json'), 'THORACIC_FRACTIONS')

    print("Extracting LUMBAR_FRACTIONS...")
    js = extract_block(lines, r'const LUMBAR_FRACTIONS\s*=', start_line_hint=3896)
    parse_and_write(js, os.path.join(CONFIG_DIR, 'skeleton', 'lumbar_fractions.json'), 'LUMBAR_FRACTIONS')

    # ──────────────────────────────────────────────
    # 13. Organs, vascular, brain
    # ──────────────────────────────────────────────
    other_arrays = [
        ('ORGAN_DEFS', 1568, 'organs.json'),
        ('VASCULAR_DEFS', 1622, 'vascular.json'),
        ('BRAIN_DEFS', 1674, 'brain.json'),
    ]
    for varname, hint, filename in other_arrays:
        print(f"Extracting {varname}...")
        js = extract_block(lines, rf'const {varname}\s*=', start_line_hint=hint)
        parse_and_write(js, os.path.join(CONFIG_DIR, filename), varname)

    # ──────────────────────────────────────────────
    # 14. MESH_DATA (line 381 — huge single-line object)
    # ──────────────────────────────────────────────
    print("\nExtracting MESH_DATA (this may take a moment)...")
    js_mesh = extract_mesh_data(lines)
    json_mesh = js_to_json(js_mesh)
    try:
        mesh_data = json.loads(json_mesh)
    except json.JSONDecodeError as e:
        print(f"  ERROR parsing MESH_DATA: {e}")
        debug_path = os.path.join(MESHDATA_DIR, 'mesh_data.json.debug')
        with open(debug_path, 'w') as f:
            f.write(json_mesh[:5000])
            f.write("\n...\n")
            # Write around the error position
            pos = e.pos if hasattr(e, 'pos') else 0
            f.write(f"\n--- Around position {pos} ---\n")
            f.write(json_mesh[max(0, pos - 200):pos + 200])
        print(f"  Debug text written to {debug_path}")
        sys.exit(1)

    # Split MESH_DATA into separate files by top-level key
    # Expected keys: skull group names (cranium, jaw, teeth, etc.), face data
    print(f"  MESH_DATA has {len(mesh_data)} top-level keys: {list(mesh_data.keys())[:10]}...")

    # Write the full mesh_data as one file
    outpath = os.path.join(MESHDATA_DIR, 'mesh_data.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(mesh_data, f, separators=(',', ':'))  # compact to save space
    size_mb = os.path.getsize(outpath) / (1024 * 1024)
    print(f"  Wrote {outpath} ({size_mb:.2f} MB)")

    # Also split into individual files for each top-level key
    for key, value in mesh_data.items():
        safe_key = key.replace(' ', '_').replace('/', '_')
        key_path = os.path.join(MESHDATA_DIR, f'{safe_key}.json')
        with open(key_path, 'w', encoding='utf-8') as f:
            json.dump(value, f, separators=(',', ':'))
        print(f"  Wrote {key_path}")

    print("\nExtraction complete!")


if __name__ == '__main__':
    main()
