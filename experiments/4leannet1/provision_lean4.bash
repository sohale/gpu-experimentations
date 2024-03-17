set -ex

# from: https://lean-lang.org/lean4/doc/make/ubuntu.html
# Please ensure you have the following build tools available and then follow the generic build instructions.
# https://lean-lang.org/lean4/doc/make/index.html

sudo apt-get update
sudo apt-get install git libgmp-dev cmake ccache clang

git clone https://github.com/leanprover/lean4 --recurse-submodules
cd lean4

git config submodule.recurse true


mkdir -p build/release
cd build/release
cmake ../..
make

