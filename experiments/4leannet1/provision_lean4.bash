set -ex

# Don't use this. Use elan instead. From:
# https://leanprover-community.github.io/install/linux.html
# https://lean-lang.org/lean4/doc/dev/index.html#dev-setup-using-elan
# curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- --default-toolchain none
# source $HOME/.elan/env
# elan default stable
# elan completions bash > /etc/bash_completion.d/elan.bash-completion
#
# what is this about?
# code lean.code-workspace


# Pulls & builds & installs latest Lean4

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

# works
echo "make complete"
