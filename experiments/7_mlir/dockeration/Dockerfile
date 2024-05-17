
# distro: "ubuntu"


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

ARG NPROC=2
ARG WORK_DIR=/workdir
WORKDIR ${WORK_DIR}

# RUN apt-get update && apt-get install -y apt-utils

# ENV DEBIAN_FRONTEND=noninteractive
# lean first layer:
# RUN apt-get clean -y && \
#    rm -rf /var/lib/apt/lists/*

# Install basic tools
RUN echo "First layer: updated apt-get (non-lean)" \
    && distro=$(cat /etc/os-release|grep -Po '(?<=^ID=").*(?=")|(?<=^ID=)[^"].*[^"]') \
    && TZ="Europe/London" \
    && { [ "${distro}" = "debian" ] || [ "${distro}" = "ubuntu" ]; } \
    \
    && DEBIAN_FRONTEND=noninteractive && \
          apt-get update && \
          apt-get install -y --no-install-recommends tzdata && \
          ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && echo "Pre-installation layer created (dev)" \
    || (echo "Unsupported distribution: ${distro}. Only 'ubuntu' or 'debian' are supported. Breaking the build" \
        && exit 3)
# End of distribution-specific part
RUN : && \
          apt-get install -qq -y --no-install-recommends \
                  autoconf automake ca-certificates clang cmake cppcheck \
                  curl default-jdk-headless gcc g++ git libncurses-dev \
                  libtool make maven ninja-build openjdk-11-jdk-headless \
                  python3 python3-dev python3-distutils python3-numpy \
                  python3-pip python3-pytest-xdist python3-setuptools \
                  python3-typing-extensions zlib1g-dev && \
          rm -rf /var/lib/apt/lists/* && \
          ln -sf /usr/bin/pytest-3 /usr/bin/pytest \
          \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --upgrade pip

# WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

# Install protobuf
ARG PROTOBUF_VERSION=21.12
RUN git clone -b v${PROTOBUF_VERSION} --recursive https://github.com/protocolbuffers/protobuf.git \
    && cd protobuf && ./autogen.sh \
    && ./configure --enable-static=no \
    && make -j${NPROC} install && ldconfig \
    && cd python && python3 setup.py install --cpp_implementation \
    && cd ../.. && rm -rf protobuf

# Install jsoniter
ARG JSONITER_VERSION=0.9.23
RUN JSONITER_URL=https://repo1.maven.org/maven2/com/jsoniter/jsoniter/${JSONITER_VERSION} \
    && JSONITER_FILE=jsoniter-${JSONITER_VERSION}.jar \
    && curl -s ${JSONITER_URL}/${JSONITER_FILE} -o /usr/share/java/${JSONITER_FILE}

# Clone and build llvm-project and run tests
ARG BUILD_SHARED_LIBS=OFF
RUN git clone -n https://github.com/llvm/llvm-project.git \
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
    && cmake --build . --parallel ${NPROC} -- ${MAKEFLAGS} \
    && (cmake --build . --parallel ${NPROC} --target check-mlir || \
        [ "$(uname -m)" = "s390x" ]) \
    && rm -rf /tmp/* \
    && echo "llvm-project commit $(git rev-parse HEAD) successfully built"

LABEL llvm_project_successfully_built=yes


# https://github.com/sohale/neopiler/blob/62cfe86487ee9e78923578a339bec1675e704d28/toolchain/dockering/conan_with_clang.Dockerfile
# https://github.com/onnx/onnx-mlir/blob/main/docker/Dockerfile.llvm-project
# as of: https://github.com/onnx/onnx-mlir/blob/c992bd8d05c964e8c0a5fcfcbf911ca81ec4af71/docker/Dockerfile.llvm-project
