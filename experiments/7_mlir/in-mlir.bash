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

# these are paths in the host system ^ (URLs from that view)

# MLIR_BINARY_DIR will bt the actual bin (for that version)
# todo: also can be exported as a label by docker, and into an ENV

#    MLIR_BIN-->MLIR_BASE??="\mlir" # Docker's ARG BUILD_WORK_DIR, ENV $WORK_DIR
#    MLIR_USER is docker's user
#     --env MLIR_HOME="/home/myuser" \

#   --volume "/home/ephemssss/novorender/oda-sdk":"/home/ephemssss/novorender/oda-sdk" \
#    --workdir "$(pwd)" \

docker run \
    --interactive --tty --rm \
    \
    --net=host \
    \
    --env BASE_PATH="${BASE_PATH}" \
    --env _initial_host_cwd="$(pwd)" \
    --env HOST_HOME="${HOME}" \
    --env GIT_REPO_ROOT="${GIT_REPO_ROOT}" \
    --volume "$GIT_REPO_ROOT":"$GIT_REPO_ROOT" \
    --volume "$WORK":"$WORK" \
    --workdir "${WORK}" \
    \
    --env MLIR_BASE="/mlir" \
    --env MLIR_USER="myuser" \
    \
    ${MLIR_IMAGE_NAME}  \
    \
    bash -c "$(cat <<'EOF_STARTUP'
      echo "You are inside docker."
      set -eux

      # todo: add this to docker as ENV
      # export PS1='temp: MLIR > ' # in case it breaks

      echo "You are: $HOME"
      nproc

      # It all works undeer this assumption:
      # todo: assert "/home/${MLIR_USER}" == $HOME

      export DEBIAN_FRONTEND=noninteractive
      # not necessary, but useful for debugging & dev
      # sudo apt update && \
      sudo apt install \
            debconf dialog \
            apt-utils \
            strace \
            \
            tmux git \
            -y
      # move pipx to here, or a later layer?
      # unzip expect default-jdk pre-commit tree
      # ncdu
      # neovim
      # wdiff
      # if windows: mingw-w64 winehq-stable winbind
      # if xwindows: xorg libx11-6  x11-apps xterm  #no: x11vnc xvfb
      # if GPU: nvidia-cuda-toolkit
      # if nwtwork: whois net-tools squid apache2-utils

      # Then, continue interactively
      unset DEBIAN_FRONTEND
      set +x  # echo off

      echo "✨🔌🏛️🏁🏁☆✴︎❂⭐︎☞✻❋✼❉✱❁ ℳ - 𝓜𝓛𝓘𝓡〖〗⎛⎞⎜⎟«»♞♘☃︎☀︎☼☁︎⛅︎🂡"
      echo
      echo "       ⛅︎"
      echo "✨⭐︎ You are inside docker ⭐︎✨"
      echo

      # source $GIT_REPO_ROOT/experiments/7_mlir/ansi_colors.env.bash
      source $GIT_REPO_ROOT/provisioning_scripts/includes/ansi_colors.env.bash

      # already there:
      # pipx, pip3, python3, python3-pip, python3-venv
      # jdk, gcc, g++, ninja-build, make, python3-numpy, pytest
      # jq, cppcheck, clang
      # protobuf, jsoniter (java?)
      # todo: mojo
      # potentially usegul: go
      # ispirational: zig, spark, rust, clangd, neovim, ANTLR, Conan,

      echo "« Key paths: »"
      # export PATH="$PATH:/opt/msvc/bin/$DESIRED_ARCH"
      # echo "Added to you path: /opt/msvc/bin/$DESIRED_ARCH"
      echo "Docker (MLIR)'s HOME=$HOME"
      echo "HOST_HOME=$HOST_HOME"
      echo "GIT_REPO_ROOT=$GIT_REPO_ROOT"
      echo "_initial_host_cwd=$_initial_host_cwd"

      echo -n 'realpath ~  : '
      realpath ~
      realpath ~/.bashrc
      ls -alth ~/.bashrc || :

      # /mlir/llvm-project/build/bin/mlir-opt --version
      # export MLIR_BINARY_DIR=/mlir/llvm-project/build/bin
      export MLIR_BINARY_DIR=$MLIR_BASE/llvm-project/build/bin
      $MLIR_BINARY_DIR/mlir-opt --version

      echo "MLIR_BASE=${MLIR_BASE}"
      echo "MLIR_BINARY_DIR=${MLIR_BINARY_DIR}"
      export B="${MLIR_BINARY_DIR}"
      echo 'B=$MLIR_BINARY_DIR'

      echo "« Key commands: »"
      echo -en "$BLUE_PALE"
      # echo "./scripts/inside_msvc-wine/compile3.bash" >>~/.bash_history
      echo "/mlir/llvm-project/build/bin/mlir-opt --version" >>~/.bash_history
      echo "$MLIR_BINARY_DIR/mlir-opt --version" >>~/.bash_history
      echo '$MLIR_BINARY_DIR/mlir-opt --version' >>~/.bash_history
      echo '$MLIR_BINARY_DIR/mlir-opt ' >>~/.bash_history
      echo '$B/mlir-opt ' >>~/.bash_history
      cat  ~/.bash_history
      echo -en "$COLOR_RESET"


      #  >> /home/myuser/.bashrc  == $HOME/.bashrc ==   ~/.bashrc
      cat <<-'__________BASHRC__________' >> ~/.bashrc

         # PROMPT_COMMAND='err=$?; if [[ $err -ne 0 ]]; then _ps1_my_error="\\[\\033[0;31m\\]🔴 $err\[\033[00m\]"; else _ps1_my_error=""; fi'
         PROMPT_COMMAND='{ __exit_code=$?; if [[ $__exit_code -ne 0 ]]; then _ps1_my_error="🔴${__exit_code}"; else _ps1_my_error=""; fi; }'

         export PS4=" 🗣️  "
         # \[\033[01;33m\]$(cut -c1-12 /proc/1/cpuset)
         echo "inside .bashrc    \$\$=$$"
         export PS1='\[\033[01;33m\]𝓜𝓛𝓘𝓡 \[\033[00;34m\]container:@\h \[\033[01;34m\]\w\[\033[00m\]\n\[\033[01;32m\]$(whoami)\[\033[00m\]  \[\033[00;31m\]${_ps1_my_error}\[\033[01;32m\] \$ \[\033[00m\]'


__________BASHRC__________
      #

      exec bash   # --norc --noprofile
      # check /home/myuser/.bashrc
      cat /home/myuser/.bashrc

      echo "Exiting exec bash-c inside docker"
      echo "Exiting docker"
EOF_STARTUP
)"

# [1] Based on github.com/sohale/ifc2brep-0/scripts/wine_init_sol3.bash
