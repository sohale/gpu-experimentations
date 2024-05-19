#!/bin/bash

set -eux

# cd ~/gpu-experimentations/experiments/7_mlir

docker images

# source env_common.env

export BASE_PATH="/home/ephemssss/gpu-experimentations/experiments/7_mlir"
export MLIR_IMAGE_NAME="mlir-dev"
# The work done inside MLIR. Rename: MLIR_WORK
export WORK="/home/ephemssss/gpu-experimentations/experiments/7_mlir"
# for sake of git commands
export GIT_REPO_ROOT="/home/ephemssss/gpu-experimentations"

#    MLIR_BIN="\mlir" # Docker's ARG BUILD_WORK_DIR, ENV $WORK_DIR

#   --volume "/home/ephemssss/novorender/oda-sdk":"/home/ephemssss/novorender/oda-sdk" \
#    --workdir "$(pwd)" \

docker run \
    --interactive --tty --rm \
    \
    --net=host \
    \
    --env BASE_PATH="${BASE_PATH}" \
    --env _initial_cwd="$(pwd)" \
    --env HOST_HOME="$HOME" \
    --volume "$GIT_REPO_ROOT":"$GIT_REPO_ROOT" \
    --volume "$WORK":"$WORK" \
    --workdir "${WORK}" \
    \
    --env MLIR_BIN="\mlir" \
    \
    ${MLIR_IMAGE_NAME}  \
    \
    bash -c "$(cat <<'EOF_STARTUP'
      echo "You are inside docker."
      echo "You are: $HOME"

      export DEBIAN_FRONTEND=noninteractive
      # not necessary, but very useful for debugging xwindows connection
      apt update \
         && apt install \
            debconf dialog \
            apt-utils \
            strace \
            \
            tmux git \
            -y

      # Then, continue interactively
      unset DEBIAN_FRONTEND

      echo "✨🔌🏛️🏁🏁☆✴︎❂⭐︎☞✻❋✼❉✱❁ℳ𝓜𝓛𝓘𝓡〖〗⎛⎞⎜⎟«»♞♘☃︎☀︎☼☁︎⛅︎🂡You are inside docker."

      source ansi_colors.env.bash

      # already there:
      # pipx, pip3, python3, python3-pip, python3-venv
      # jdk, gcc, g++, ninja-build, make, python3-numpy, pytest
      # jq, cppcheck, clang
      # protobuf, jsoniter (java?)
      # todo: mojo
      # potentially usegul: go
      # ispirational: zig, spark, rust, clangd, neovim, ANTLR, Conan,

      echo "« Key paths: »"
      export PATH="$PATH:/opt/msvc/bin/$DESIRED_ARCH"
      echo "Added to you path: /opt/msvc/bin/$DESIRED_ARCH"
      echo "Docker's HOME=$HOME"
      echo "HOST_HOME=$HOST_HOME"
      echo "GIT_REPO_ROOT=$GIT_REPO_ROOT"
      echo "_initial_cwd=$_initial_cwd"


      echo "« Key commands: »"
      echo "./scripts/inside_msvc-wine/compile1.bash" >>~/.bash_history
      echo "./scripts/inside_msvc-wine/compile2.bash" >>~/.bash_history
      echo "./scripts/inside_msvc-wine/compile3.bash" >>~/.bash_history
      cat  ~/.bash_history

      export PS1='\[\033[01;36m\]container\[\033[00m\]:\[\033[01;35m\]@\h 𝓜𝓛𝓘𝓡\[\033[01;34m\]\w\[\033[00m\]\n\[\033[01;32m\]$(whoami) \[\033[00m\] \[\033[01;33m\]$(cut -c1-12 /proc/1/cpuset)\[\033[01;32m\] \$ \[\033[00m\]'
      exec bash   # --norc --noprofile

      echo "Exiting exec bash-c inside docker"
      echo "Exiting docker"
EOF_STARTUP
)"

# [1] Based on github.com/sohale/ifc2brep-0/scripts/wine_init_sol3.bash
