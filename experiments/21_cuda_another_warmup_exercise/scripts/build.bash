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


# Optional: enable experimental C++26 if supported
CXX26_OPT=""
if [[ "${USE_CPP26:-0}" == "1" ]]; then
  CXX26_OPT="-DUSE_CPP26=ON"
  echo "Enabling experimental C++26 (via -std=c++2c if supported)"
fi
# CMake Warning at CMakeLists.txt:24 (message):
#  C++26 flag -std=c++2c not supported by this compiler.  Using C++23.




# Configure and build
cmake -S . -B build $CXX26_OPT
cmake --build build -j"$(nproc)"

: || {
# Optional: run the built executable
if [[ -x "build/cpp_test1" ]]; then
  echo "Running cpp_test1..."
  ./build/cpp_test1 || true
fi
}
