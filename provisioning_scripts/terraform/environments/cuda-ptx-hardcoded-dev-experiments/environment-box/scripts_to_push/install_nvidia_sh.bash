#
# Install dev tools, especially GPU-related toos: Mainly: nvidia tools
# can be docker, or, install commands.

# for github cli (enables the ssh)
# source $SCRIPTS_BASE_REMOTE/refresh_ssh_agent.env

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


#python
#>>> print(torch.cuda.is_available())
#>>> import torch
#True

TAG=25.01-py3-igpu
# docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:xx.xx-py3

# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
