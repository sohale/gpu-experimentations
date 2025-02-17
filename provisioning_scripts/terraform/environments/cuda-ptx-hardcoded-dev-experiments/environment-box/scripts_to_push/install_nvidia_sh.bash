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
# Ensure: `nvidia-docker2` is installed i.e. "NVIDIA Container Toolkit"
dpkg -l | grep -i nvidia-docker
{
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list && \
    sudo apt update && sudo apt install -y nvidia-docker2

    sudo systemctl restart docker
}
# Ensure: NVIDIA Runtime is Enabled
docker info | grep -i runtime


# CUDA, Ubuntu 22.04, x86_64
# see: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

: ||{
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
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

DCITAG="25.01-py3"
DCINAME="nvcr.io/nvidia/pytorch"
DCHOME="/home/mine"

docker --version
docker pull "$DCNAME:$TAG"

docker run \
        --gpus all \
        -it --rm \
        --volume "$HOME/oggi":"$DCHOME/oggi"\
        --volume "$HOME/scripts-sosi:$DCHOME/scripts-sosi"\
        --volume "$HOME/secrets":"$DCHOME/secrets"\
        --volume "$HOME/work":"$DCHOME/work"\
        "$DCINAME:$DCITAG"
