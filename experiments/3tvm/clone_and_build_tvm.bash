# If necessary, First set up a VPS using https://gist.github.com/sohale/9112b718625e372b1d407f91f2011296

set -ex

cd ~/gpu-experimentations/experiments/3tvm

# Installing (building) from source:
# Following https://tvm.apache.org/docs/install/from_source.html

: """skip
# git clone --recursive https://github.com/apache/tvm tvm
"""

: """skip
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
"""

cd ~/gpu-experimentations/experiments/3tvm/tvm

mkdir -p build
cd build

# skip
echo ||
cp ../cmake/config.cmake ../build

# I want
# set(USE_NVTX ON)

: """
< set(USE_NVTX OFF)
> set(USE_NVTX ON)


< set(USE_LLVM OFF)
> set(USE_LLVM ON)

< set(USE_MLIR OFF)
> set(USE_MLIR ON)
"""

export TVM_LOG_DEBUG="ir/transform.cc=1,relay/ir/transform.cc=1"

# Insall latest cmake & ninja
: """ skip
lsb_release -a

# https://apt.kitware.com/
sudo apt-get install ca-certificates gpg wget

test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update

test -f /usr/share/doc/kitware-archive-keyring/copyright || sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg

sudo apt-get install kitware-archive-keyring

echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy-rc main' | sudo tee -a /etc/apt/sources.list.d/kitware.list >/dev/null

sudo apt-get update

sudo apt-get install cmake

echo

sudo apt install ninja-build
"""

cmake --version  # 3.29.0-rc2
ninja --version  # 1.10.1

echo

# cd build
ls config.cmake

# Now, let's build:
cmake .. -G Ninja
