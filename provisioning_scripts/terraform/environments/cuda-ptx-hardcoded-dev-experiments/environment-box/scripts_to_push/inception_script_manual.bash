# This is the script you are supposd to run manually in the remote machine.
# This is pu tthe as a reuls tof the whole provisioning (+following bash("show_results") )
#   todo: rename the "show_results" step, or create a "show_reault" too? it is perhaps "user_results" which is too vague.
# we are in the remote machine

# environment_boxes/neurotalk/scripts_to_push/docker_user_fix.bash
# docker_user_fix.bash
# Differnt to  local_manual__setup_at_creation.bash . That is from direct scripts
#
echo -e "\n${BASH_SOURCE[0]}\n$(date)" >> ~/.sosi-footprints.log



    set -eux
    echo v0.0.5
    #repeated

    # sudo usermod -aG docker $USER
    # The newgrp command is used to change the current group ID during a session. This subshell disrupts the script's execution flow.
    # newgrp docker || :  # idempotent


    # Add the user to the docker group if not already a member
    # if ! groups $USER | grep -q "\bdocker\b"; then
    if : ; then
        sudo usermod -aG docker $USER
        echo "User $USER added to the docker group."

        groups $USER
        groups

        # SWITCHING

        # Old Approach: (skipped)
        : || { # skipped

                # avoid 'newgrp', use 'sg' instead.
                # For scripting, use 'sg' because it doesn't start an interactive shell.

                # Run newgrp docker to change the current group ID
                echo "Changing to docker group. Please re-login for the group change to take effect."
                # newgrp docker  # this alone, spawns a new interactive shell
                newgrp docker <<EOF
                set -eu
                echo "Current User: $(whoami)"
                echo "User ID (UID): $(id -u)"
                echo "Group ID (GID): $(id -g)"
                echo "All Groups: $(id -Gn)"
                echo "Group Details: $(id)"
                # docker ps
EOF
                # Exit the script with a message
                # echo "Please re-run the script after logging out and back in."
                # exit 1  # no need to exist 1 with error anymore. Since we don't use the interactive

                # but it will need restart
                exec bash # but this will start a new shell, and the script will not continue

        }

        # New approach:

        sg docker -c '
        echo "Group switched!" ;
        echo "Now inside the new group \"docker\"." ;

        echo "Current User: $(whoami)" ;
        echo "User ID (UID): $(id -u)" ;
        echo "Group ID (GID): $(id -g)" ;
        echo "All Groups: $(id -Gn)" ;
        echo "Group Details: $(id)" ;

        echo "Running docker ps" ;
        docker ps
        '
    fi


    groups
    # Still fails first time:
    groups | grep "docker"
    groups $USER | grep -q "\bdocker\b"
    # sudo systemctl status docker
    # sudo systemctl start docker


    # todo: move to:
    #  terraform/environments/neurotalk/scrupts_to_push/system_hardware_spec_info.bash

    # I am Remote.

    # injected during this provisioning (not tfplly, but after that: show_resutls)
    # SCRIPTS_PLACED -> SCRIPTS_BASE -> SCRIPTS_BASE_REMOTE
    SCRIPTS_BASE_REMOTE="/home/paperspace/scripts-sosi"
    # must be already there, since we need environment_boxes/neurotalk/scripts_to_push to have been copied there
    mkdir -p "$SCRIPTS_BASE_REMOTE"


    # going to run: environment_boxes/neurotalk/scripts_to_push/system_hardware_spec_info.bash
    # no, "map"  "environment_boxes/neurotalk/scripts_to_push/" to "TARGET_SCRIPTS_DIR" (the version of it in terraform/common/localmachine/upload_scripts_to_there.bash)
    # TARGET_SCRIPTS_DIR__
    TARGET_SCRIPTS2PUSH_DIR__="$SCRIPTS_BASE_REMOTE/scripts_to_push"
    # here, TARGET means remote, dont call it target
    # REMOTE_SCRIPTS_DIR: name ( it is already there)
    # TARGET_SCRIPTS_DIR: verb ( it is going to be there)
    # predictive assignment: it is gonna be its reference, BEFORE dedfining it!!
    # another differece with otmasl referemvdes
    # separate source & separate starts
    # sudo environment_boxes/neurotalk/scripts_to_push/system_hardware_spec_info.bash
    sudo bash "$TARGET_SCRIPTS2PUSH_DIR__/system_hardware_spec_info.bash"
    # these commands have travelled ...



    # target is the internal name of the sane thuibg (referne0/)

    OGGI="~/oggi"
    mkdir -p ~/oggi




    # environment_boxes/neurotalk/scripts_to_push/ghcli-install.bash

    # the following, perhaps, whould not be int he same file. But, in some symbolic (static-analysis-time: SA-Time) represeantation, they should follow eachother.
    bash "$TARGET_SCRIPTS2PUSH_DIR__/ghcli-install.bash"
    bash "$TARGET_SCRIPTS2PUSH_DIR__/ghcli-login.bash"

    # This script applies refresh_ssh_agent.env, but it shall be done in startup script for session (.bashrc) or manually. (currently, manually)
    # Since `refresh_ssh_agent.env` is used in two places, I am saving it in a file.
    source $SCRIPTS_BASE_REMOTE/refresh_ssh_agent.env
    {
    gh --version
    gh auth status
    } # || :

    #ODDI_PATHJ=
    cd ~/oggi
    { ls -alth ~/oggi/gpu-experimentations && echo "already cloned"; } || \
    # git clone git@github.com:sohale/pocs_for_nikolai.git
    #  https://github.com/sohale/pocs_for_nikolai.git
    git clone git@github.com:sohale/gpu-experimentations.git

    # docker does not need be here, unless we want to work strictly within Docker
    #    I think the idea was that everything to be done inside the docker, to be one safe side regarding installging NVidia tools
    #    But it will not keep the history.
    #    Maybe I should just share the history (with docker host, i.e. TF's remote GPU machine)
    #         & mount the volumes, etc?
    cd ~/oggi
    R="$(realpath .)"

    # this script itself, needs run something manually, now, it asks the user to do another manual step! (by meanual, I mean "in interactive mode")
    set +x # echo off
    echo "You need to do this manually: ✋nual exports: You need to do this manually in interactive (shell/bash terminal mode) on remote machine: (which in tuen may have more manual steps)

        # this script that just ran: (but run twice)
✋      bash ~/scripts-sosi/scripts_to_push/inception_script_manual.bash

✋      source /home/paperspace/scripts-sosi/refresh_ssh_agent.env
✋      bash ~/scripts-sosi/scripts_to_push/install_nvidia_sh.bash
✋      docker run -it --rm -v $R:$R -w $R/ nvcr.io/nvidia/pytorch:22.02-py3

or:
      R=$R
      "'docker run -it --rm -v $R:$R -w $R/ nvcr.io/nvidia/pytorch:22.02-py3'"
      "


: '
   (more) handy scripts:


    pspace machine start $(cd environments/cuda-ptx-hardcoded-dev-experiments ; terraform output -raw instance_id)

'