
# syntax=docker/dockerfile:1.3
# Necessary for new syntax suh as ARGs to be used in FROM

# Latest (estimatted) image size: ~ 10.3GB

ARG BASE_IMAGE="ubuntu:jammy"
FROM ${BASE_IMAGE}

# Label the image for various checking and cleanup
ARG LLVM_PROJECT_SHA1
ARG LLVM_PROJECT_SHA1_DATE
ARG LLVM_PROJECT_DOCKERFILE_SHA1
# ARG ONNX_MLIR_PR_NUMBER
# ARG ONNX_MLIR_PR_NUMBER2
# ARG MLIR_PR_NUMBER
LABEL llvm_project_sha1=${LLVM_PROJECT_SHA1}
LABEL llvm_project_sha1_date=${LLVM_PROJECT_SHA1_DATE}
LABEL llvm_project_dockerfile_sha1=${LLVM_PROJECT_DOCKERFILE_SHA1}
# LABEL onnx_mlir_pr_number=${ONNX_MLIR_PR_NUMBER}
# LABEL onnx_mlir_pr_number2=${ONNX_MLIR_PR_NUMBER2}


# Standard dockerhub labels
LABEL maintainer="Sohail Siadat <sohale@gmail.com>"
LABEL version="0.01"
LABEL description="A MLIR build including MLIR development environment"


# ARG: Only build time, not available at runtime. not preserved in the final image.
# ENV: Only runtime, not available at build time. (during execusion of Docker image). persist (in the final image too?).
#   is set (being assgned to) (evaluated) at build time(? no! but if RHG is an ARG??), readable in run (executable) time (& build time).
# LABEL: metadata, only in Docker image, for informational & organizational purposes.

ARG NPROC=1
ARG WORK_DIR=/workdir
WORKDIR ${WORK_DIR}

# We don't want this at runtime
ARG TZ="Europe/London"

# RUN apt-get update && apt-get install -y apt-utils

ENV DEBIAN_FRONTEND=noninteractive

# I don't want "cache busting".
# In fact I want to save build time (by reusing layers that haven't changed) rather than save space.
# To clean/prune the cache: docker builder prune --all --force

# Install basic tools: non-lean approach (apt-get update stays in the Docker layer)
# $distro will be: "ubuntu"
RUN : \
    && echo "First layer: updated apt-get" \
    && set -eux \
    \
    && distro=$(cat /etc/os-release|grep -Po '(?<=^ID=").*(?=")|(?<=^ID=)[^"].*[^"]') \
    && { [ "${distro}" = "debian" ] || [ "${distro}" = "ubuntu" ]; } \
    \
    && apt-get update  -y \
    && apt-get install -y \
           apt-utils \
           ca-certificates curl wget \
    \
    && TZ="Europe/London" \
    # tzdata: tzdata is the package that contains time zone and daylight-saving time dat
    && apt-get install -y tzdata \
    && ls -alth /usr/share/zoneinfo/ \
    # assertion
    && ls -alth /usr/share/zoneinfo/${TZ} \
    && ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    # ^ Necessary too?
    && dpkg-reconfigure -f noninteractive tzdata \
    \
    # Not sure if needed or not
    && apt-get autoremove -y \
    # Make it a thin first layer
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Pre-installation layer created (dev)" \
    || ( : \
      && echo "Unsupported distribution: ${distro}. Only 'ubuntu' or 'debian' are supported. Breaking the build" \
      && exit 3 \
    )
# For lean first layer, append to `RUN` :
#    && apt-get clean -y \
#    && rm -rf /var/lib/apt/lists/* \

# --no-install-recommends
# study: `-qq -y --no-install-recommends`
#
# apt-get install
#     -qq   # quiet (verbosity)
#     -y (use with DEBIAN_FRONTEND=noninteractive )
#     --no-install-recommends  # Only install the explicily mentioned packages, not the recommended ones
#    --no-cache # Do not cache the package files
#    && apt-get update --no-cache # No such thing?
#
# apt-get clean -y
# Clean the package cache. Removes the local repository of retrieved package files. his frees up space by deleting .deb files from /var/cache/apt/archives.
# clean up the local repository of retrieved package files:
# deletes /deb files in /var/cache/apt/archives/ and /var/cache/apt/archives/partial/
# i.e. actual downloaded packages that were installed on the system
# metadata about what packages are available for installation is not deleted
# actual packages (downloaded .deb files) are deleted, but the packages remain installed and functional

# rm -rf /var/lib/apt/lists/*
# reduce the image size after installing packages.
# deletes information about what packages are available for installation.
# no longer needed once the packages have been installed.
# metadata about what packages are deleted
# actual packages (downaded files) are not deleted, again, the packages remain installed and functional

# package "lists" ( "metadata") are updated, in /var/lib/apt/lists/
# you can also --reinstall some package ".deb" files: /var/cache/apt/archives/
# to downgrade:
# dpkg -i /var/cache/apt/archives/older_version_package.deb
# du -sh /var/cache/apt/archives/

# End of base 1 part

RUN : \
   && set -eux \
   && apt-get update  -y \
   && :

# End of the "hydrated" part

# dev layer [2]
ARG DevUser="myuser"
ENV \
   force_color_prompt=yes \
   color_prompt=yes

# Dev tools
RUN : \
   && set -eux \
   \
   # Note that I already have had curl, wget here, but it's OK to have it again
   && apt-get install -y curl bash \
   && apt-get install -y sudo wget \
   # Also enable `--ssh default` in your `DOCKER_BUILDKIT=1 docker build` command: \
   #    That will enable `RUN --mount=type=ssh ...` to work \
   && apt-get install -y git  ssh-client ca-certificates patch less \
   && useradd -m -s /bin/bash -G sudo ${DevUser} \
   && echo "${DevUser} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
   && echo 'Username will be: ${DevUser} ' \
   \
   && :
# TODO: Leave the bash / interactive / user bit to the last layers
# Switch to the new user "after" the user is created
# USER ${DevUser}

# ENV PATH=${APP_HOME}/bin:${PATH}




# ARG ARCH_DEFAULT=$(uname -m)
# ENV ARCH=${ARCH:-$ARCH_DEFAULT}
# RUN echo "Architecture at build time: $ARCH, available in execution time"
# RUN echo "Architecture at build time: $ARCH_DEFAULT"

# Will be deferred to runtime
ARG ARCH_BUILD_TIME=$(uname -m)
ENV ARCH_RUNTIME=${ARCH_RUNTIME:-$ARCH_BUILD_TIME}
RUN : \
   && echo  "Architecture at build time: $ARCH_RUNTIME (ARCH_RUNTIME), available in execution time" \
   && echo  "Architecture at build time: $ARCH_BUILD_TIME (ARCH_BUILD_TIME)" \
   && :
# Architecture at build time: $(uname (ARCH_RUNTIME), available in execution time
# Architecture at build time: $(uname (ARCH_BUILD_TIME)

# A reusable python layer (reusable)
RUN : \
   && sudo apt-get install -y pipx \
   && :

# Use --no-install-recommends to make things very tightly efficient
RUN : \
   && set -eux \
   && apt-get install \
            -y --no-install-recommends \
         autoconf automake clang cmake cppcheck \
         default-jdk-headless gcc g++ libncurses-dev \
         libtool make maven ninja-build openjdk-11-jdk-headless \
         python3 python3-dev python3-distutils python3-numpy \
         python3-pip python3-pytest-xdist python3-setuptools \
         python3-typing-extensions zlib1g-dev \
   # && rm -rf /var/lib/apt/lists/* \
   && ln -sf /usr/bin/pytest-3 /usr/bin/pytest \
   \
   && ln -sf /usr/bin/python3 /usr/bin/python \
   && pip3 install --upgrade pip \
   && :
# WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

# A few more cute & useful tools: jq xxd
RUN : \
   && set -eux \
   && apt-get install -y \
         jq  \
   && jq --version \
   :


# Install protobuf
ARG PROTOBUF_VERSION=21.12
RUN : \
    && set -eux \
    && git clone -b v${PROTOBUF_VERSION} --recursive https://github.com/protocolbuffers/protobuf.git \
    && cd protobuf && ./autogen.sh \
    && ./configure --enable-static=no \
    && make -j${NPROC} install && ldconfig \
    && cd python && python3 setup.py install --cpp_implementation \
    && cd ../.. && rm -rf protobuf \
    && :

# Install jsoniter
ARG JSONITER_VERSION=0.9.23
RUN : \
    && JSONITER_URL=https://repo1.maven.org/maven2/com/jsoniter/jsoniter/${JSONITER_VERSION} \
    && JSONITER_FILE=jsoniter-${JSONITER_VERSION}.jar \
    && curl -s ${JSONITER_URL}/${JSONITER_FILE} -o /usr/share/java/${JSONITER_FILE} \
    && :

# Clone to save llvm-project in a seaprate folder to save time
# On new realases, this needs to be done from scratch
# Note: This is in root user
# Note: WORKDIR is used
RUN : \
   && echo "cloning -n into: $(pwd), and ARG WORKDIR=${WORKDIR}" \
   && git clone -n https://github.com/llvm/llvm-project.git \
   :

# After clone, build llvm-project and run tests
ARG BUILD_SHARED_LIBS=OFF

RUN : \
    && cd llvm-project \
    && git checkout ${LLVM_PROJECT_SHA1} \
    && mkdir -p build && cd build \
    # Build with clang since gcc on ppc64le doesn't support __builtin_thread_pointer
    && CC=clang CXX=clang++ \
         cmake -G Ninja ../llvm \
             -DLLVM_ENABLE_PROJECTS=mlir \
             -DLLVM_TARGETS_TO_BUILD="host" \
             -DCMAKE_BUILD_TYPE=Release \
             -DLLVM_ENABLE_ASSERTIONS=ON \
             -DLLVM_ENABLE_RTTI=ON \
             -DLLVM_ENABLE_LIBEDIT=OFF \
             -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
   \
   && cmake --build . --parallel ${NPROC} -- ${MAKEFLAGS} \
   && ( \
        cmake --build . --parallel ${NPROC} --target check-mlir \
        || [ "$(uname -m)" = "s390x" ]  \
      ) \
   && rm -rf /tmp/* \
   && echo "llvm-project commit $(git rev-parse HEAD) successfully built"

# $(uname -m) ---> "x86_64"
#                  "s390x":  IBM Z machine architecture

ENV DEBIAN_FRONTEND=

USER ${DevUser}
# Will use the USER ${DevUser}
CMD ["bash"]

# LABEL add metadata to a Docker image
# For CI/CD criteria and Quaiity Gates (Verification) (ready for deployment) (debugging & build status)
LABEL llvm_project_successfully_built=yes

# https://github.com/sohale/neopiler/blob/62cfe86487ee9e78923578a339bec1675e704d28/toolchain/dockering/conan_with_clang.Dockerfile
# https://github.com/onnx/onnx-mlir/blob/main/docker/Dockerfile.llvm-project
# as of: https://github.com/onnx/onnx-mlir/blob/c992bd8d05c964e8c0a5fcfcbf911ca81ec4af71/docker/Dockerfile.llvm-project
#
# [2] See dev layer in https://github.com/sohale/neopiler/blob/62cfe86487ee9e78923578a339bec1675e704d28/toolchain/dockering/conan_with_clang.Dockerfile#L37
