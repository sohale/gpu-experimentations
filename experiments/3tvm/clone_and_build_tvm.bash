# If necessary, First set up a VPS using https://gist.github.com/sohale/9112b718625e372b1d407f91f2011296

set -ex

cd ~/gpu-experimentations/experiments

# Installing (building) from source:
# Following https://tvm.apache.org/docs/install/from_source.html

# skip
# git clone --recursive https://github.com/apache/tvm tvm


# sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

cd ~/gpu-experimentations/experiments/tvm

mkdir build
cd build

cp ../cmake/config.cmake ../build
