#!/usr/bin/env bash
#
# build-engine.sh -- Build a standalone srclight binary using PyInstaller.
#
# Creates a clean venv, installs srclight + pyinstaller, runs the spec file,
# and packages the output into a tarball (Linux/macOS) or zip (Windows/MSYS).
#
# Usage:
#   ./packaging/pyinstaller/build-engine.sh
#
# Environment variables:
#   SRCLIGHT_EXTRAS  - pip extras to install (default: "docs,pdf")
#   PYTHON           - python interpreter to use (default: python3)
#   SKIP_VENV        - set to 1 to skip venv creation (use current env)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPEC_FILE="$SCRIPT_DIR/srclight.spec"

PYTHON="${PYTHON:-python3}"
SRCLIGHT_EXTRAS="${SRCLIGHT_EXTRAS:-docs,pdf}"
SKIP_VENV="${SKIP_VENV:-0}"

BUILD_DIR="$REPO_ROOT/build/pyinstaller"
DIST_DIR="$REPO_ROOT/dist"
VENV_DIR="$BUILD_DIR/venv"

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)  PLATFORM="linux"  ;;
    Darwin*) PLATFORM="macos"  ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
    *)       PLATFORM="unknown" ;;
esac

echo "=== srclight PyInstaller build ==="
echo "  Platform:  $PLATFORM ($ARCH)"
echo "  Python:    $PYTHON"
echo "  Extras:    $SRCLIGHT_EXTRAS"
echo "  Repo root: $REPO_ROOT"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Create build venv (unless SKIP_VENV=1)
# ---------------------------------------------------------------------------
if [ "$SKIP_VENV" = "1" ]; then
    echo "--- Skipping venv creation (SKIP_VENV=1) ---"
    PIP="pip"
    PYINSTALLER="pyinstaller"
else
    echo "--- Creating build venv at $VENV_DIR ---"
    mkdir -p "$BUILD_DIR"
    "$PYTHON" -m venv "$VENV_DIR"

    # Activate
    if [ "$PLATFORM" = "windows" ]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi

    PIP="pip"
    PYINSTALLER="pyinstaller"
fi

# ---------------------------------------------------------------------------
# Step 2: Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Installing srclight[$SRCLIGHT_EXTRAS] + pyinstaller ---"
$PIP install --upgrade pip setuptools wheel 2>&1 | tail -1
$PIP install pyinstaller 2>&1 | tail -1

if [ -n "$SRCLIGHT_EXTRAS" ]; then
    $PIP install -e "$REPO_ROOT[$SRCLIGHT_EXTRAS]" 2>&1 | tail -1
else
    $PIP install -e "$REPO_ROOT" 2>&1 | tail -1
fi

echo "  Installed srclight $(python -c 'import srclight; print(srclight.__version__)')"
echo "  Installed PyInstaller $($PYINSTALLER --version)"

# ---------------------------------------------------------------------------
# Step 3: Run PyInstaller
# ---------------------------------------------------------------------------
echo ""
echo "--- Running PyInstaller ---"
$PYINSTALLER \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR/work" \
    --clean \
    --noconfirm \
    "$SPEC_FILE"

# Verify the binary works
BINARY="$DIST_DIR/srclight/srclight"
if [ "$PLATFORM" = "windows" ]; then
    BINARY="$DIST_DIR/srclight/srclight.exe"
fi

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi

echo ""
echo "--- Verifying binary ---"
"$BINARY" --version

# ---------------------------------------------------------------------------
# Step 4: Package the output
# ---------------------------------------------------------------------------
echo ""
echo "--- Packaging ---"

VERSION=$("$BINARY" --version 2>&1 | grep -oP '[\d]+\.[\d]+\.[\d]+' || echo "unknown")

if [ "$PLATFORM" = "windows" ]; then
    ARTIFACT="$DIST_DIR/srclight-${VERSION}-${PLATFORM}-${ARCH}.zip"
    (cd "$DIST_DIR" && zip -qr "$ARTIFACT" srclight/)
else
    ARTIFACT="$DIST_DIR/srclight-${VERSION}-${PLATFORM}-${ARCH}.tar.gz"
    tar -czf "$ARTIFACT" -C "$DIST_DIR" srclight/
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ARTIFACT_SIZE=$(du -sh "$ARTIFACT" | cut -f1)
BINARY_SIZE=$(du -sh "$BINARY" | cut -f1)

echo ""
echo "=== Build complete ==="
echo "  Binary:   $BINARY ($BINARY_SIZE)"
echo "  Archive:  $ARTIFACT ($ARTIFACT_SIZE)"
echo "  Version:  $VERSION"
echo ""
echo "To test:"
echo "  $BINARY serve --web"
echo ""
echo "To distribute, ship: $ARTIFACT"
