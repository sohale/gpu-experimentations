
* context

NVIDIA  TU104GL [Quadro RTX 4000] (rev a1)
Ubuntu 22

To install: cuda & clang
* CUDA: as cuda-toolkit-13-1
* Clang: as 21.1.7


Starting point of installation of CUDA

* Also see: See [scripts/session_history.txt]

https://github.com/NVIDIA/apt-packaging-cuda-keyring


Install gh, pythonnn, t
```bash
sudo apt install python3.10-venv

```
### Clang

```bash
cd /opt
sudo mkdir llvm-21
sudo chown $USER:$USER llvm-21
cd llvm-21
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.7/LLVM-21.1.7-Linux-X64.tar.xz
tar -xf ./LLVM-21.1.7-Linux-X64.tar.xz --strip-components=1
echo 'export PATH=/opt/llvm-21/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
exit
clang++ --version

# Missing step: needed for standard library
sudo apt install libstdc++-12-dev
# verify compiling
clang++ -std=c++17 -stdlib=libstdc++  cpp_test1.cpp

```

### CUDA

```bash
lsb_release -a
lspci

sudo apt-get update
sudo apt install -y wget gnupg ca-certificates
sudo apt-get install -y debhelper devscripts dpkg-dev make

git clone git@github.com:NVIDIA/apt-packaging-cuda-keyring.git
cd apt-packaging-cuda-keyring/

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-archive-keyring.gpg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin cuda.pin
cat cuda-archive-keyring.gpg | gpg --dearmor > ./cuda-archive.keyring.gpg
cat ./cuda.pin
# check three files:
ls -alth  ./build.sh ./cuda.pin ./cuda-archive-keyring.gpg

DISTRO=ubuntu2204 ARCH=x86_64 ./build.sh  ./cuda-archive-keyring.gpg ./cuda.pin

sudo apt-get update

ls cuda-keyring*_all.deb
sudo dpkg --install cuda-keyring*_all.deb
# ^ installed cuda-keyring_1.1-1_all.deb

apt-cache search cuda-toolkit
# chise the cuda-toolkit version. => 13-1
sudo apt install cuda-toolkit-13-1

# not yet
nvidia-smi
```
