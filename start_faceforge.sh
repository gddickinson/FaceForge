#!/bin/zsh
# Start the full FaceForge GUI.
#
# Usage:  ./start_faceforge.sh
#
# Picks the interpreter that has FaceForge's dependencies (PySide6, PyOpenGL):
#   1. $FACEFORGE_PYTHON if set,
#   2. the flika conda environment used for development,
#   3. python3 on the PATH.
# Runs from the project folder so the src/ layout and assets resolve.

set -e
cd "$(dirname "$0")"

if [ -n "$FACEFORGE_PYTHON" ]; then
  PY="$FACEFORGE_PYTHON"
elif [ -x /opt/anaconda3/envs/flika/bin/python ]; then
  PY=/opt/anaconda3/envs/flika/bin/python
else
  PY="$(command -v python3)"
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m faceforge.app "$@"
