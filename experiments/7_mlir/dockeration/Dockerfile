



ARG BASE_IMAGE="ubuntu:jammy"
FROM ${BASE_IMAGE}

# Label the image for various checking and cleanup
ARG LLVM_PROJECT_SHA1
ARG LLVM_PROJECT_SHA1_DATE
ARG LLVM_PROJECT_DOCKERFILE_SHA1
ARG ONNX_MLIR_PR_NUMBER
ARG ONNX_MLIR_PR_NUMBER2
LABEL llvm_project_sha1=${LLVM_PROJECT_SHA1}
LABEL llvm_project_sha1_date=${LLVM_PROJECT_SHA1_DATE}
LABEL llvm_project_dockerfile_sha1=${LLVM_PROJECT_DOCKERFILE_SHA1}
LABEL onnx_mlir_pr_number=${ONNX_MLIR_PR_NUMBER}
LABEL onnx_mlir_pr_number2=${ONNX_MLIR_PR_NUMBER2}

ARG NPROC=1
ARG WORK_DIR=/workdir
WORKDIR ${WORK_DIR}


ENV TZ=Europe/London

# RUN apt-get update && apt-get install -y apt-utils

# ENV DEBIAN_FRONTEND=noninteractive

# I don't want "cache busting".
# In fact I want to save build time (by reusing layers that haven't changed) rather than save space.
# To clean/prune the cache: docker builder prune --all --force

# Install basic tools: non-lean approach (apt-get update stays in the Docker layer)
# $distro will be: "ubuntu"
RUN : \
    && echo "First layer: updated apt-get" \
    && distro=$(cat /etc/os-release|grep -Po '(?<=^ID=").*(?=")|(?<=^ID=)[^"].*[^"]') \
    && { [ "${distro}" = "debian" ] || [ "${distro}" = "ubuntu" ]; } \
    \
    && DEBIAN_FRONTEND=noninteractive \
    && apt-get update  -y \
    && apt-get install -y \
           apt-utils \
           ca-certificates curl wget bash \
    \
    && TZ="Europe/London" \
    # tzdata: tzdata is the package that contains time zone and daylight-saving time dat
    && apt-get install -y tzdata \
    && ls -alth /usr/share/zoneinfo/ \
    && ls -alth /usr/share/zoneinfo/${TZ} \
    && ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    # ^ Necessary too?
    && dpkg-reconfigure -f noninteractive tzdata \
    \
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

RUN : \
   && apt-get update  -y \
   && :
# package "lists" ( "metadata") are updated, in /var/lib/apt/lists/
# you can also --reinstall some package ".deb" files: /var/cache/apt/archives/
# to downgrade:
# dpkg -i /var/cache/apt/archives/older_version_package.deb
# du -sh /var/cache/apt/archives/

# End of distribution-specific part

# Use --no-install-recommends to make things very tight
RUN : \
   && apt-get install \
            -qq -y --no-install-recommends \
         autoconf automake clang cmake cppcheck \
         default-jdk-headless gcc g++ git libncurses-dev \
         libtool make maven ninja-build openjdk-11-jdk-headless \
         python3 python3-dev python3-distutils python3-numpy \
         python3-pip python3-pytest-xdist python3-setuptools \
         python3-typing-extensions zlib1g-dev \
   # && rm -rf /var/lib/apt/lists/* \
   && ln -sf /usr/bin/pytest-3 /usr/bin/pytest \
   \
   && ln -sf /usr/bin/python3 /usr/bin/python \
   && pip3 install --upgrade pip
# WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

# Install protobuf
ARG PROTOBUF_VERSION=21.12
RUN : \
    && git clone -b v${PROTOBUF_VERSION} --recursive https://github.com/protocolbuffers/protobuf.git \
    && cd protobuf && ./autogen.sh \
    && ./configure --enable-static=no \
    && make -j${NPROC} install && ldconfig \
    && cd python && python3 setup.py install --cpp_implementation \
    && cd ../.. && rm -rf protobuf

# Install jsoniter
ARG JSONITER_VERSION=0.9.23
RUN : \
    && JSONITER_URL=https://repo1.maven.org/maven2/com/jsoniter/jsoniter/${JSONITER_VERSION} \
    && JSONITER_FILE=jsoniter-${JSONITER_VERSION}.jar \
    && curl -s ${JSONITER_URL}/${JSONITER_FILE} -o /usr/share/java/${JSONITER_FILE}

# Clone and build llvm-project and run tests
ARG BUILD_SHARED_LIBS=OFF
RUN : \
    && git clone -n https://github.com/llvm/llvm-project.git \
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

# $(uname -m) ---> x86_64

ENV DEBIAN_FRONTEND=

CMD ["bash"]

# LABEL add metadata to a Docker image
# For CI/CD criteria and Quaiity Gates (Verification) (ready for deployment) (debugging & build status)
LABEL llvm_project_successfully_built=yes

# https://github.com/sohale/neopiler/blob/62cfe86487ee9e78923578a339bec1675e704d28/toolchain/dockering/conan_with_clang.Dockerfile
# https://github.com/onnx/onnx-mlir/blob/main/docker/Dockerfile.llvm-project
# as of: https://github.com/onnx/onnx-mlir/blob/c992bd8d05c964e8c0a5fcfcbf911ca81ec4af71/docker/Dockerfile.llvm-project
