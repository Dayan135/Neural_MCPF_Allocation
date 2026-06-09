#!/usr/bin/env bash
#
# Set up the third-party RobustMCPF solver used to generate ground-truth labels.
#
# RobustMCPF is NOT vendored in this repo (it is large, ships solver binaries, and
# has its own upstream with no license). This script reproduces the exact working
# copy the project expects:
#   1. clone upstream at a pinned commit
#   2. compile the LKH-3 TSP solver for the local platform
#   3. apply our one-line BasicMAPF patch (scripts/basic_mapf.patch)
#
# Note: BasicMAPF only needs LKH. GLKH (the orientation-aware E-GTSP solver) is
# not built here because this project never uses the RCbssEff path.
#
# Usage:  ./scripts/setup_robustmcpf.sh
set -euo pipefail

REPO_URL="https://github.com/yehonatan280198/RobustMCPF.git"
PINNED_COMMIT="b858767d75bc8f7e494936b2732a5db24e388f7a"   # AAAI-2026 release, 2026-06-04

# Resolve paths relative to this script so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEST="$PROJECT_DIR/RobustMCPF"
PATCH="$SCRIPT_DIR/basic_mapf.patch"
LKH_DIR="$DEST/LKH-3.0.11"

# 1. Clone at the pinned commit ------------------------------------------------
if [ -d "$DEST/.git" ]; then
    echo ">> RobustMCPF git checkout already present, skipping clone"
elif [ -e "$DEST" ]; then
    echo "ERROR: $DEST exists but is not a git checkout."
    echo "       An unmanaged copy is already here (e.g. the original vendored files)."
    echo "       If it works, you don't need this script. To re-provision from scratch,"
    echo "       remove it first:  rm -rf '$DEST'"
    exit 1
else
    echo ">> Cloning RobustMCPF into $DEST"
    git clone "$REPO_URL" "$DEST"
    git -C "$DEST" checkout --quiet "$PINNED_COMMIT"
fi

# 2. Compile LKH-3 for the local platform --------------------------------------
# Upstream tracks a prebuilt LKH binary, but it is platform-specific (e.g. a Linux
# build won't run on macOS). Always do a clean rebuild so the binary is native.
echo ">> Building LKH-3 for the local platform"
make -C "$LKH_DIR" clean >/dev/null 2>&1 || true
make -C "$LKH_DIR"

# 3. Apply our BasicMAPF patch (idempotent) ------------------------------------
if git -C "$DEST" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo ">> BasicMAPF patch already applied, skipping"
else
    echo ">> Applying BasicMAPF patch"
    git -C "$DEST" apply "$PATCH"
fi

echo ">> RobustMCPF setup complete."
