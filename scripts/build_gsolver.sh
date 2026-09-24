#!/bin/bash
# Configure and build gsolver (library, C++ examples, Python bindings) on this host.
#
# Usage: scripts/build_gsolver.sh <build_dir> [jobs]
#
# Host-specific settings (see reports/phase0.md):
#  - yaml-cpp 0.8.0 is not installed system-wide; a static, PIC build lives in
#    $YAML_PREFIX. Its include directory must be added with -isystem and its lib
#    directory with -L for executables *and* the Python shared module.
#  - CMake's FindPython prefers /usr/bin/python3.12, which has no headers here;
#    the system python3 (3.9) has them, so it is pinned explicitly.
set -euo pipefail
B=${1:?usage: $0 <build_dir> [jobs]}
J=${2:-48}
YAML_PREFIX=${YAML_PREFIX:-/home/os717/gsolver-deps/install}
REPO=$(cd "$(dirname "$0")/.." && pwd)
L="-L$YAML_PREFIX/lib64"
cmake -S "$REPO" -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DGSOLVER_BUILD_EXAMPLES=ON -DGSOLVER_BUILD_PYTHON=ON \
  -DCMAKE_PREFIX_PATH="$YAML_PREFIX" \
  "-DCMAKE_CXX_FLAGS=-isystem $YAML_PREFIX/include" \
  -DCMAKE_EXE_LINKER_FLAGS="$L" -DCMAKE_MODULE_LINKER_FLAGS="$L" -DCMAKE_SHARED_LINKER_FLAGS="$L" \
  -DPython_EXECUTABLE=/usr/bin/python3 -DGSOLVER_BUILD_CERTIFY=${GSOLVER_BUILD_CERTIFY:-ON}
cmake --build "$B" -j"$J"
