
* context

NVIDIA  TU104GL [Quadro RTX 4000] (rev a1)
Ubuntu 22

To install: cuda & clang
* CUDA: as cuda-toolkit-13-1
* Clang: as 21.1.7

### Get the GPU

Get the GPU Machine: RTX4000


#### On local machine:
Append to `~/.ssh/config`:
```txt
Host paperspace-gpu-quicktemp
    HostName 111.111.111.111
    User paperspace
    IdentityFile ~/.ssh/id_ed111111
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Client/local machine:
```bash
ssh paperspace-gpu-quicktemp
```
On VScode: `F1` -> `Remote-SSH: Connect to Host...` -> `paperspace-gpu-quicktemp`

## On the GPU machine


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
Starting point of installation of CUDA is three downloaded files:
* `build.sh`, via  `git clone` of: from https://github.com/NVIDIA/apt-packaging-cuda-keyring
* cuda-archive-keyring.gpg (via `wget`)
* cuda-ubuntu2204.pin (via `wget`)

Also see: See [scripts/session_history.txt]


```bash
lsb_release -a # Ubuntu 22.04.5 LTS
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

ls /usr/local/cuda-13.1/

echo '
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
' >> ~/.bashrc
source ~/.bashrc # or: exit

nvcc --version
#              works
nvidia-detector
#              works: output: nvidia-driver-590
# based on that output:
sudo apt install nvidia-utils-590
#              Installs: libnvidia-compute-590 nvidia-firmware-590-590.48.01 nvidia-kernel-common-590
#              suggests also: nvidia-driver-590

# Driver missing:
lsmod | grep nvidia ; ls /dev | grep nvidia

# finally: sudo apt install nvidia-driver-590

sudo apt install nvidia-driver-590
# restart
sudo reboot

# works only after restart
nvidia-smi

# verify
lsmod | grep nvidia ; ls /dev | grep nvidia
```

#### Some outputs:

```bash
lsmod | grep nvidia
```
```txt
nvidia_uvm           1777664  0
nvidia_drm            110592  0
nvidia_modeset       1495040  1 nvidia_drm
nvidia              99459072  2 nvidia_uvm,nvidia_modeset
drm_kms_helper        311296  6 bochs,drm_vram_helper,nvidia_drm
drm                   622592  8 drm_kms_helper,bochs,drm_vram_helper,nvidia,drm_ttm_helper,nvidia_drm,ttm
```

```bash
ls /dev | grep nvidia
```

```txt
nvidia0
nvidia-caps
nvidiactl
nvidia-modeset
nvidia-uvm
nvidia-uvm-tools
```

```bash
nvidia-smi
```

```txt
Wed Feb  4 20:29:59 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro RTX 4000                Off |   00000000:00:05.0 Off |                  N/A |
| 30%   35C    P8              9W /  125W |       1MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

```bash
nvidia-smi ; nvcc --version ; clang++ --version ; clang++ -std=c++17 -stdlib=libstdc++  cpp_test1.cpp
```

For workspace:
```bash
bash ./scripts/dev.bash
```
Then, on vscode, click on `exper21.generated.code-workspace`
