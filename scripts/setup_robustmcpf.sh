#!/usr/bin/env bash
#
# Build the LKH-3 TSP solver for the local platform.
#
# RobustMCPF is now vendored directly in this repo, with the BasicMAPF patch
# already applied (scripts/basic_mapf.patch kept for reference).
# This script only needs to compile the LKH binary for the current platform.
#
# Usage:  ./scripts/setup_robustmcpf.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LKH_DIR="$PROJECT_DIR/RobustMCPF/LKH-3.0.11"

echo ">> Building LKH-3 for the local platform"
mkdir -p "$LKH_DIR/SRC/OBJ"
make -C "$LKH_DIR" clean >/dev/null 2>&1 || true
make -C "$LKH_DIR"
echo ">> LKH build complete: $LKH_DIR/LKH"
