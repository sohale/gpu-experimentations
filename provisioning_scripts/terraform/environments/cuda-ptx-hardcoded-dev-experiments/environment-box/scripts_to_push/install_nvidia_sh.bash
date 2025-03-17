#
# Install dev tools, especially GPU-related toos: Mainly: nvidia tools
# can be docker, or, install commands.

# for github cli (enables the ssh)
# source $SCRIPTS_BASE_REMOTE/refresh_ssh_agent.env

function experim1 {
docker --version


# docker pull nvcr.io/nvidia/pytorch:22.02-py3
# docker pull nvcr.io/nvidia/pytorch:25.01-py3-igpu

# docker pull nvcr.io/nvidia/pytorch:25.01-py3-igpu
docker pull nvcr.io/nvidia/pytorch:25.01-py3
# 12.59 GB   # 01/27/2025 4:59 PM, 2 Architectures

# inspect/fetch  other docker tags for nvcr.io/nvidia/pytorch:25.01-py3-igpu
# Fetch and list other docker tags for nvcr.io/nvidia/pytorch:25.01-py3-igpu
curl -s https://registry.hub.docker.com/v1/repositories/nvcr.io/nvidia/pytorch/tags | jq -r '.[].name'
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags

# --gpus all


#python
#>>> print(torch.cuda.is_available())
#>>> import torch
#True

# TAG=25.01-py3-igpu
# docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:xx.xx-py3

# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

}


function install_inplace {
    # non-containerized. Not preferred.

    # legacy: Ubuntu 22 's own:
    # sudo apt install nvcc

    # NVidia's own installation instrucitons: (subject to update in the future)
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-8
    nvcc --version

}

# check foundations:
# Ensure: NVIDIA GPU is Detected
nvidia-smi
# Ensure: NVIDIA Runtime is Enabled
docker info | grep -i runtime

# Critical step:
# Ensure: `nvidia-docker2` is installed i.e. "NVIDIA Container Toolkit"
dpkg -l | grep -i nvidia-docker

function install_nvidia_docker
{
   # deprecated solution
   #  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
   # curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && \
   # curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list && \
   # sudo apt update && sudo apt install -y nvidia-docker2
   # sudo systemctl restart docker

   # todo: add (move) to the current apt-based installations script

    mkdir -p ~/workspace
    cd ~/workspace
    # Enables "nvidia-driver-570"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt install ubuntu-drivers-common
    sudo ubuntu-drivers devices
    sudo ubuntu-drivers devices | sort
    # For A5000?
    sudo apt install nvidia-driver-570
    # may need: sudo reboot now
    # nvidia-smi
    # KEY:
    sudo apt install nvidia-container-toolkit
    # Will need: (?)
    #       sudo reboot now

    # sudo apt-get -y install cuda-toolkit-12-8
    # sudo systemctl restart docker

    # Test / verify:

    # test the "--gpus all"
    docker run --rm --gpus all nvcr.io/nvidia/cuda nvidia-smi

    # Should be non-empty:
    dpkg -l | grep -i nvidia-docker

}


# CUDA, Ubuntu 22.04, x86_64
# see: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network


: ||{
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    # no, see above instead:
    sudo apt-get -y install cuda-toolkit-12-8
            # ^ Many interesting things ...
            # cuda-opencl
            # libcusolver
            # libcurand
            # libcusparse
            # cuda-nvrtc
            # cuda-nvdisasm
            # cuda-cccl
            # cuda-cupti
            # cuda-nsight-compute
            # ...
    # maybe: sudo apt install nvidia-cuda-toolkit

    # sudo apt-get install -y nvidia-open
}
# GPT-noise: # https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/


: || {
    # Deepter debugging
    # incorrect image: nvidia/cuda:12.0-base
    # small:
    docker run --rm --gpus all nvcr.io/nvidia/cuda nvidia-smi

    # or:
    docker run --rm --gpus all nvidia/cuda:11.4.3-base-ubuntu20.04 nvidia-smi

    # causes this error:
    # docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
    # GPT: The following drivers are available: nvidia, nvidia-container-runtime.
}


# Some cli arg are because of the following warning:
#   " NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
#   "    insufficient for PyTorch.  NVIDIA recommends the use of the following flags:
#   "    docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...


# note: v3


DCITAG="25.01-py3"
DCINAME="nvcr.io/nvidia/pytorch"
# DCHOME="/home/ubuntu"   # The username for NGC (nvidia/pytorch).
DCHOME="/root"
DCWORKSPACE="/workspace"

docker --version
docker pull "$DCNAME:$TAG"

#         --volume "$HOME/oggi":"$DCWORKSPACE/oggi"\
# -e PS1="\u@\h:\w\$ "

# maybe: sudo apt install nvidia-cuda-toolkit


#         -it --rm \
# bad:     -t \

docker run \
        --gpus all \
        -it  \
        \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        \
        --volume "$HOME/scripts-sosi:$DCHOME/scripts-sosi"\
        --volume "$HOME/secrets":"$DCHOME/secrets"\
        --volume "$HOME/workspace":"$DCWORKSPACE"\
        \
        --env PROMPT_COMMAND='{ __exit_code=$?; if [[ $__exit_code -ne 0 ]]; then _ps1_my_error="${__exit_code} 🔴"; else _ps1_my_error=""; fi; }' \
        --env _PS1='\[\033[01;33m\]🐳➫  𝗚𝗣𝗨 \[\033[00;34m\]container:@\h \[\033[01;34m\]\w\[\033[00m\]\n➫ \[\033[01;32m\]$(whoami)\[\033[00m\]  \[\033[00;31m\]${_ps1_my_error}\[\033[01;32m\] \$ \[\033[00m\]' \
        \
        "$DCINAME:$DCITAG"

        # Mixes session with the whole computer! The session overrides the PS1
        # Can you runit, and then, "code" into it?

# # instrucitons: Three computers:
# 1. within docker:
# curl -fsSL https://code-server.dev/install.sh | sh
#  # code-server
# # sudo systemctl enable --now code-server@$USER

#   above inside th docker
#   ssh paperspace@74.82.28.88 in the macos client
#   local: Remote-Containers: Attach to Running Container


# local is not a good name for TF. TRue local may be another computer.

# on host:
# Name: Dev Containers
# Id: ms-vscode-remote.remote-containers
# Description: Open any folder or repository inside a Docker container and take advantage of Visual Studio Code's full feature set.
# Version: 0.397.0
# Publisher: Microsoft
# VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
