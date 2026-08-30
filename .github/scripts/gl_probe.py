#!/usr/bin/env python3
"""Report whether a headless GL context can be acquired on this machine.

This is the experiment the `golden-images` CI job exists to run.

The reasoning: ``tools/glcontext.py`` reaches Apple's software rasteriser
through CGL with ``kCGLPFARendererID = kCGLRendererGenericFloatID``, which
works precisely *because* it does not ask the window server to enumerate
displays.  A GitHub macOS runner is headless with no window server, which is
the same situation.  If a context comes up there, golden-image rendering and
the CPU-vs-driver shader agreement tests can gate every push instead of being
a local-only step.

That has not been tried on a GitHub runner.  This script is how the answer gets
recorded: it prints the renderer banner on success and the precise CGL error on
failure, so the job log answers the question either way.

Exit codes: 0 context acquired, 1 no context (with the reason), 2 the module
could not even be imported.  The job is ``continue-on-error`` until the first
real run settles it.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"platform: {platform.platform()}  python: {sys.version.split()[0]}")
    try:
        from tools.glcontext import GLContextError, acquire_offscreen_gl
    except Exception as exc:                                        # noqa: BLE001
        print(f"cannot import tools.glcontext: {exc!r}", file=sys.stderr)
        return 2

    for prefer in ("hardware", "software", "auto"):
        try:
            info = acquire_offscreen_gl(prefer)
        except GLContextError as exc:
            print(f"\nprefer={prefer!r}: NO CONTEXT — {exc}")
            continue
        except Exception as exc:                                    # noqa: BLE001
            print(f"\nprefer={prefer!r}: unexpected {type(exc).__name__}: {exc}")
            continue
        print(f"\nprefer={prefer!r}: acquired")
        print(info.banner())
        for attempt in info.attempts:
            print(f"  attempt: {attempt}")
        print(
            "\nRESULT: a headless GL context is available on this runner.\n"
            "The shader/GPU-agreement tests can run here, and golden-image\n"
            "capture becomes possible as soon as the runner also has the\n"
            "BodyParts3D dataset (tools/capture_golden.py renders 16 named\n"
            "FMA meshes and cannot synthesise them)."
        )
        return 0

    print(
        "\nRESULT: no GL context on this runner by any path.\n"
        "Golden-image validation stays a local step.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
