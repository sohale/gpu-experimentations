# syntax=docker/dockerfile:1.4
# ^ Don't change


# Leaving .github/docker/ub_pyenv.dockerfile.v1.deprecated (see notes)
#   for deprecating/expiring an old concern used in that file:
#    Now: not separating dev concerns as docker layers. Hence, combining the apt-get steps etc.
#    Also, adopt more of concern: [layering-related] size efficiency.
# More decisions:
#    * Use sudo, root, freely.
#    * Use pyenv, not python3. (i.e. not system python3

# build script:
# docker buildx build --secret id=date,src=./sosi_sfile_1.txt  --tag ghcr.io/sohale/ub-pyenv-image:latest -f ./ub_pyenv.dockerfile --push ./ctx/

# Used this in .github/workflows/ci_asic_19.yaml
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

# It will not replace inside of echo '...' commands!
# So, practically, an ARG is an env?
#   Perhaps: ARG is an ENV but only during installations.
#   But only ENV is passed at `docker run`.
# adds \" instead of the `"` for the echo.
# It has to be after the installation sudo, also.
# Dependency chain:
#  installation of sudo -> actual ceration of user (needs sudo) -> setting USER -> runing any command using "user"



# Special (reserved) vars:
#      `WORKDIR`

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
#    date > ./sosi_sfile_1.txt
#    docker buildx build --secret id=date,src=./sosi_sfile_1.txt  --tag test-ubpyenv1:latest -f ./ub_pyenv.dockerfile --load ./ctx/

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

# If you want to customise the username for CMD/bash:
ARG ARG_DEV_USER="devuser"

# User who installed pyenv:
ARG ARG_USER="pyenvuser"
ARG ARG_UGROUP="pyenvgroup"

# compromise: maje it one-more-nested
ARG ARG_BASEPATH="/opt/pyenvu/pyenv"
# ARG ARG_BASEPATH="/root/.pyenv" # Not good: user-specific installation.

# This ENV is like an ARG:
ENV \
    PYENV_ROOT="$ARG_BASEPATH" \
    MY_PYTHON_VERSION="${ARG_MY_PYTHON_VERSION}"

# Is this a good practice?
ENV \
    PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Implemenation stuff (not parameters of the produced image):
# ARG ARG_BUILD_WORK_DIR=/mlir
# WORKDIR ${ARG_BUILD_WORK_DIR}
# ENV ENV_WORK_DIR=${ARG_BUILD_WORK_DIR}

# Notes:
#  "git checkout" does not fetch from the interwebs.

# The ...`_local.txt` file is created during the build, but, in the build, we are noot "INSIDE" the image (??).
#. but where is the "./" ? I deliverately used it.


# Start with non-dev mode for apt-get commands:
ENV \
    DEBIAN_FRONTEND=noninteractive

RUN : \
    && echo "First layer: 1.0: apt-get update" \
    && apt-get update

RUN : \
    && echo "First layer: 1.1: apt-get" \
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
          wget \
# the dev tools
          sudo \
          ssh-client ca-certificates patch \
          bash curl wget git patch \
          less \
    && :
# ^ Notes:
#   * llvm: Is llvm needed? (for pyenv)
#   * git: "git" is needed for pyenv, which I happen to use in non-dev layer.



# requires: git
# attempt to get some commit hash, for debugging purposes. It is not important, so I skip it.
#RUN : \
#    && touch ./sosi_commit_hash_local.txt || : \
#    && echo "checkout and build: pwd=$(pwd), ARG WORKDIR=${WORKDIR}" \
#    && cd dev \
#    && ls -alth \
#    && pwd \
#    && git rev-parse HEAD > ./sosi_commit_hash_local.txt \
#    && echo
#
# ls -alth \. # will only show ... just that file that was touch ed.
# cwd is in '/' (also the WORKDIR). WORKDIR="", empty.

# Create a pyenv-specific user without sudo ability (unpriviledged user)
RUN : \
    && useradd -m -s /bin/bash -G sudo ${ARG_USER} \
    && groupadd ${ARG_UGROUP} \
    && usermod -aG ${ARG_UGROUP} ${ARG_USER} \
    && :
    # not a sudoer, unless proven necessary.
    # && echo "${ARG_DEV_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
RUN : \
    # create the parent: will work by the minor compromise: of nesting it one level (otherwise) unnecessarily.
    && PPPARENT="$(dirname "$ARG_BASEPATH")" \
    && mkdir -p "$PPPARENT" \
    && chown -R  $ARG_USER:$ARG_UGROUP $PPPARENT \
    && ls -alth  /opt/pyenvu \
    && :
# remove the extra-nested one?
# while still as root: create more users:

# Create a dev user "with" sudo ability
RUN : \
    && useradd -m -s /bin/bash -G sudo ${ARG_DEV_USER} \
    && echo "${ARG_DEV_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && usermod -aG ${ARG_UGROUP} ${ARG_DEV_USER} \
    && echo "Username ${ARG_DEV_USER} created, can sudo, can pyenv, but is not set." \
    && :


USER ${ARG_USER}


# Install pyenv
# Causion: the "pyenv install" is very slow. I have used "Python-3.12.3".
RUN : \
    && echo "Layer 2: pyenv" \
    && whoami \
    && ls -alth /opt/pyenvu \
    && curl https://pyenv.run | bash \
    && pyenv install ${ARG_MY_PYTHON_VERSION} \
    && pyenv global ${ARG_MY_PYTHON_VERSION} \
    && pyenv rehash \
    && python --version \
#    && :
#
# Can be slow:
#   A non-critical step that may fail. FIXME:
#     pyenv update  || echo "err code: $?"  || :
#   If already updated, the "pyenv update" will exit with an error code.
#
#  Merge, to avoid keeing both unupdated and updated
# RUN : \
    && pyenv update || : \
    && pyenv install ${MY_PYTHON_VERSION} || : \
    && pyenv global ${MY_PYTHON_VERSION} \
    && pyenv rehash \
    && whoami \
    && echo "pyenv sanity check done." \
    && :


# Development mode necessities: e.g. bash

# If dev mode (BTW, pyenv is suitable for dev mode. May not be essential for GH Actions CI mode, but let's keep it.)
# Order of USER vs RUN matters.


# Seal for security:
#  (but do it only after inalling the  rest of dev tools. In fact, for dev, we may need to have more. So, I remove it. If we want it, we need to add it after the last apt-get. But we need to cancel this anyway)
# RUN : \
#     && rm -rf /var/lib/apt/lists/*



# do it after installing pyenv.
#RUN : \
#    && groupadd ${ARG_UGROUP} \
#    && usermod -aG ${ARG_UGROUP} ${ARG_USER} \
#    && chown -R ${ARG_USER}:${ARG_UGROUP} ${ARG_BASEPATH} \
#    && find "$ARG_BASEPATH" -exec chmod g=u {} + \
#    && find ${ARG_BASEPATH} -type d -exec chmod g+rx {} \; \
#    && find ${ARG_BASEPATH} -type f -exec chmod g+r {} \; \
#    && find "$ARG_BASEPATH" -type d -exec chmod g+s {} + \
#    && usermod -aG ${ARG_UGROUP} ${ARG_DEV_USER} \
#    && :

# not, indistriminatedly:
#  chmod -R g+rx ${ARG_BASEPATH}


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
# Uses the secret with --moun
# Don't use "./"
RUN --mount=type=secret,id=date,target=/run/secrets/build_date \
    BUILD_DATE=$(cat /run/secrets/build_date) && \
    echo "Build date is ${BUILD_DATE}" && \
    cat /run/secrets/build_date && \
    echo


# If you want to customise the username:
# Does not "set" the user, just the future `RUN`s will be using this user.
# So, must be after a RUN command that creates this user! Because, if befor, that (and any) will fail.

# Must choose a user:
USER ${ARG_DEV_USER}
# USER ${ARG_USER}
# USER root

# From here on, you will need to use sudo (if you activate it) and you dont have sudo password.
# Also note that the "root" was the user for pyenv.

# Lesson: user may run it using a different user i.e. "runner" (in GH Actions CI workflow). But here we assume root.
