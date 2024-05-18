#!/bin/bash

# todo: move docker temp folder to the desired location (cloud container's volume)
# mlir_dev_docker_build.sh

set -eux


# docker version
docker --version
# Docker version 24.0.5, build 24.0.5-0ubuntu1~22.04.1

# Make sure buildx is installed
docker buildx version || \
{
   mkdir -p ~/.docker/cli-plugins;
   BUILD_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
   echo "BUILD_VERSION: ${BUILD_VERSION}"
   curl -Lo ~/.docker/cli-plugins/docker-buildx "https://github.com/docker/buildx/releases/download/${BUILD_VERSION}/buildx-${BUILD_VERSION}.linux-amd64"
   chmod +x ~/.docker/cli-plugins/docker-buildx
   docker buildx version
}

# Featuring "SSH Forwarding":
# DOCKER_BUILDKIT=1 \
docker  build \
   --ssh default \
   \
   --progress=plain \
   \
   -t mlir-v-tbc  \
   -f Dockerfile \
   .

# Also possible
   #  --secret id=secret1,src=secret1.txt \
   #  --secret id=secret2,src=secret2.txt \
