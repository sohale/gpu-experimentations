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
   # ./helpers/install_docker_buildx.bash
   mkdir -p ~/.docker/cli-plugins;
   BUILD_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
   echo "BUILD_VERSION: ${BUILD_VERSION}"
   curl -Lo ~/.docker/cli-plugins/docker-buildx "https://github.com/docker/buildx/releases/download/${BUILD_VERSION}/buildx-${BUILD_VERSION}.linux-amd64"
   chmod +x ~/.docker/cli-plugins/docker-buildx
   docker buildx version
}

export LLVM_LATEST_RELEASE="$(./fetch_latest_release_llvm.bash)"

: << COMMENT
   ARG ARCH_DEFAULT=$(uname -m)
   ARG BASE_IMAGE="ubuntu:jammy"
   ARG LLVM_PROJECT_SHA1
   ARG LLVM_PROJECT_SHA1_DATE
   ARG LLVM_PROJECT_DOCKERFILE_SHA1   # only for label

   ARG NPROC=1
   ARG WORK_DIR=/workdir
   ARG TZ="Europe/London"
   ARG DevUser="myuser"
   ARG ARCH_BUILD_TIME=$(uname -m)   # x86_64
   ARG PROTOBUF_VERSION=21.12
   ARG JSONITER_VERSION=0.9.23
   ARG BUILD_SHARED_LIBS=OFF
COMMENT


# export MLIR_IMAGE_NAME="mlir-v-tbc"
export MLIR_IMAGE_NAME="mlir-dev"

# Featuring "SSH Forwarding":
# DOCKER_BUILDKIT=1 \
docker  build \
   --ssh default \
   --progress=plain \
   \
   --build-arg LLVM_PROJECT_SHA1="$LLVM_LATEST_RELEASE" \
   \
   -t $MLIR_IMAGE_NAME  \
   -f Dockerfile \
   .

# Also possible
   #  --secret id=secret1,src=secret1.txt \
   #  --secret id=secret2,src=secret2.txt \

date
date 1>&2
