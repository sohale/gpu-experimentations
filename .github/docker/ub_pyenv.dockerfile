# syntax=docker/dockerfile:1.4
# ^ Don't change

# todo: used this in .github/workflows/ci_asic_19.yaml
# Not tested.
# Also useful for development, like GitPod. (not just GH Actions CI workflows)

# Naming: decision for places:
#    .github/docker/ub_pyenv.dockerfile
#    .github/docker/pyenv-image/Dockerfile
#    provisioning_scripts/cicd/ub_pyenv.dockerfile

# Dev Notes:

# todo: ARG, ENV, LABEL
#   Notes:
#     (A crash course in builder-x)
#           Also see: experiments/7_mlir/dockeration/Dockerfile
#
#     * ENV
#         * ENV is only runtime, not available at build time.
#         * i.e. during execution "run" of Docker image
#         * Readable in run (executable) time & build time too? (!contradiciton).
#         * ENV persists in the final image too. (via metadata? todo: check)
#         * ENV is set (being assgned to) (evaluated) at build time
#         * You cannot set it programmatically:
#               Cannot: `ENV COMMIT_HASH="$(cat ./sosi_commit_hash_local.txt)"` or `BUILD_FINISH_DATE=$(date)`
#               Because you cannot "RUN" things in an "ENV".
#         * Can inherit from the base/parent image, e.g. for the ${x:-$y} pattern (todo: check).
#
#     * ARG:
#         * ARG is only build time, not available at runtime.
#         * ARG not preserved in the final image.
#         * ARG accesses as: ${argname}, or $argname, or ${x:-$y}
#                e.g. in RUN commands, and in ENV. (expended in the cli commands in RUN)
#                can be used to parametrise ENV, LABEL.
#                and RUN.
#         * Hence, ARG can clash with ENV variables, if they have the same name (? check).
#         * ARG also can be used in LABEL values (to parametrise them)
#         * ARG is set (being assgned to) (evaluated) at build time, not at runtime.
#         * Not sure: you can ENV newenvname=${envname:-$argname}
#
#     * LABEL:
#         * Standard dockerhub labels. metadata, only in Docker image, for informational & organizational purposes only.
#         * Unlike ARG, cannot be read!
#         * cannot be read in RUN, nor in ENV!
#         * There are some standard dockerhub labels: `maintainer`, `description`.

# Order of USER,  RUN, ENV, etc matters. Even ENV can be changed between RUN steps (see DEBIAN_FRONTEND).

# To enable BuildKit: (Docker BuildKit’s extended syntax)
# First line should start with: # syntax=docker/dockerfile:1.3
#      Necessary for new syntax suh as ARGs to be used in FROM
# Must build using:
#          DOCKER_BUILDKIT=1 docker build --ssh default ...
#
#         --ssh default
#         { "features": { "buildkit": true } }

# Also enable `--ssh default`: (why? Maybe: "for SSH keys, e.g. for git clone private repos")


# GH Actions CI workflow build(x):
#   See usage of "docker/setup-buildx-action@v3" in workflow

# Local machine build:
#
# I don't want "cache busting", which is to save space.
# Instead, I may want to disable it, to save build time: by reusing built layers that haven't changed.
# To manually clean / prune the cache:
#    docker builder prune --all --force

# Instructions to install buildx on your local machine::
#{
#   # ./helpers/install_docker_buildx.bash
#   mkdir -p ~/.docker/cli-plugins;
#   BUILD_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
#   echo "BUILD_VERSION: ${BUILD_VERSION}"
#   curl -Lo ~/.docker/cli-plugins/docker-buildx "https://github.com/docker/buildx/releases/download/${BUILD_VERSION}/buildx-${BUILD_VERSION}.linux-amd64"
#   chmod +x ~/.docker/cli-plugins/docker-buildx
#   docker buildx version
#}
#
# To make sure buildx is installed
#    docker buildx version
# To actually build:
#    docker buildx build ./ub_pyenv.dockerfile

# To learn: build vs bake. (in context of "buildx")
# To learn:
#     DOCKER_BUILDKIT=1 docker build ...
#         vs
#     docker buildx ...
#         vs
#     DOCKER_BUILDKIT=1 docker build --ssh default ...

# In a buildx system:
# (remote) registry
# local images registry and local daemon
# local context directory
# local builder instances
# local builder instance (used for build)

# The driver for a "buildx-build": "--driver docker-container" vs "--driver docker" (default)
# The `--use` = current chosen (a bit like `cd` or `chroot`).
# The current (`--use`) driver can be non-buildx (unsure, check).
# You can have multiple "BUILDKIT" versions instantiated!
# A local builder instance has a "name" (via `--name`)

# More on builder:
#  "builder instance"s, configured builder instance.

#      docker buildx build \
#         --secret id=date,src=llvm_commit_hash.txt \
#         --tag myimage:latest \
#         --load \
#         -f .github/docker/ub_pyenv.dockerfile \
#         . # build context
# or:
#         --tag ghcr.io/yourname/myimage:latest
# also: --platform




# End of dev-docs.


ARG BASE_IMAGE="ubuntu:22.04"

FROM ${BASE_IMAGE}


# Standard dockerhub labels
LABEL maintainer="Sohail Siadat <sohale@gmail.com>"
LABEL description="A pyenv development environment for Yosys-based EDA/ASIC/FPGA design and verification"

ARG ARG_MY_PYTHON_VERSION="3.12.3"

# If you want to customise the username:
ARG ARG_DEV_USER="devuser"

# This ENV is like an ARG:
ENV \
    PYENV_ROOT="/root/.pyenv" \
    MY_PYTHON_VERSION="${ARG_MY_PYTHON_VERSION}"

# Is this a good practice?
ENV \
    PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"


# Notes:
#  "git checkout" does not fetch from the interwebs.

# The ...`_local.txt` file is created during the build, but, in the build, we are noot "INSIDE" the image (??).
#. but where is the "./" ? I deliverately used it.

RUN : \
    && touch > ./sosi_commit_hash_local.txt && exit 0 || : \
    && echo "checkout and build: pwd=$(pwd), ARG WORKDIR=${WORKDIR}" \
    && git rev-parse HEAD > ./sosi_commit_hash_local.txt \
    && echo

# Start with non-dev mode for apt-get commands:
ENV \
    DEBIAN_FRONTEND=noninteractive

RUN : \
    && echo "First layer: apt-get" \
    && apt-get update \
    && apt-get install -y \
          build-essential \
          ca-certificates \
          libssl-dev \
          zlib1g-dev libbz2-dev \
          libreadline-dev \
          libncurses5-dev \
          libncursesw5-dev \
          libsqlite3-dev \
          xz-utils \
          tk-dev \
          libffi-dev liblzma-dev \
          git \
          curl \
          wget
#
#      llvm
#  Is llvm needed? (for pyenv)
# "git" is needed for pyenv, which I happen to use in non-dev layer.

# Seal for security:
RUN : \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN : \
     && echo "Layer 2: pyenv" \
    && curl https://pyenv.run | bash \
    && pyenv install ${ARG_MY_PYTHON_VERSION} \
    && pyenv global ${ARG_MY_PYTHON_VERSION} \
    && pyenv rehash \
    && python --version \
    && :

# Can be slow:
#   A non-critical step that may fail. FIXME:
#     pyenv update  || echo "err code: $?"  || :

RUN \
    pyenv update \
    && pyenv install ${MY_PYTHON_VERSION} \
    && pyenv global ${MY_PYTHON_VERSION} \
    && pyenv rehash

# Development mode necessities: e.g. bash

# If dev mode (BTW, pyenv is suitable for dev mode. May not be essential for GH Actions CI mode, but let's keep it.)
# Order of USER vs RUN matters.

# If you want to customise the username:
USER ${ARG_DEV_USER}


# Dev tools
RUN : \
    && echo "Layer 3: pyenv" \
    && set -eux \
    && apt-get install -y  sudo ssh-client ca-certificates patch \
    && apt-get install -y  bash curl wget git patch \
    && apt-get install -y  less \
    && :

RUN : \
    && set -eux \
    && useradd -m -s /bin/bash -G sudo ${ARG_DEV_USER} \
    && echo "${ARG_DEV_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && echo 'Username will be: ${ARG_DEV_USER}' \
    && :



# Leave it in a nice state for development / for a dev user:
ENV \
    DEBIAN_FRONTEND=

# Implementation content
ENV \
    force_color_prompt=yes \
    color_prompt=yes

# If dev mode only.
# The following command Will use the USER ${ARG_DEV_USER}:
CMD ["bash"]

# The concept of temporary mounts during the build:
# Temporarily mounts a local file sosi_commit_hash_local.txt into the container at "/sosi_commit_hash.txt".
# mounted only inside that RUN command, and never committed into the image (unlike COPY).
# Hence, three places! (two images: temporary (not commited) and final (which has layers + buildx-hyper-layers) , apart from , obviously, the local files in the system running the buildx)
# id and target are secrets ("free parameters")
# Correction: a secret, has both "id" and "src". (and "target"?)
RUN --mount=type=secret,id=date,target=/sosi_commit_hash.txt \
    BUILD_DATE=$(cat ./sosi_commit_hash_local.txt) && \
    echo "Build date is ${BUILD_DATE}"
