#!/usr/bin/env bash
set -euo pipefail

# Build script for cpp_test1 using CMake.
# Respects pre-exported CC/CXX from ~/.bashrc; defaults to clang/clang++ if unset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default to clang/clang++ if not provided
: "${CC:=clang}"
: "${CXX:=clang++}"
export CC CXX

echo "Using CC=$CC"
echo "Using CXX=$CXX"


# failfast:
clang++ --version
clang --version
echo $CC $CXX
nvidia-smi
nvcc --version
# clang++ -std=c++17 -stdlib=libstdc++  cpp_test1.cpp
cmake --version
# Check CMake availability
if ! command -v cmake >/dev/null 2>&1; then
  echo "Error: cmake not found in PATH. Install CMake or add it to PATH (e.g., $HOME/.local/bin)." >&2
  exit 1
fi



########### The build command

# Configure and build
# cmake -S . -B build

cmake -S . -B build -DUSE_CPP26=ON

cmake --build build -j"$(nproc)"

# alternatively:
# clang++ -std=c++17 -stdlib=libstdc++  cpp_test1.cpp

: || {
# Optional: run the built executable
if [[ -x "build/cpp_test1" ]]; then
  echo "Running cpp_test1..."
  ./build/cpp_test1 || true
fi
}
