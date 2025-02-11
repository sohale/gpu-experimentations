set -eu
# run on (from) Local Machine.
# triggered by user (should be automated), via up-....bash (`show_outputs` mode).

# wrong filename
# upload_scripts_to_there.bash
# ->
# local_manuatrigger.bash
# move this:
# provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/environment-box/local_manual__setup_at_creation.bash
# merge that into this

# The upload_scripts ... should not be a file, but, a fucniton

# compare: (merge?)
#    terraform/common/localmachine/upload_scripts_to_there.bash
#    environment_boxes/neurotalk/local_manual__setup_at_creation.bash

# Why different?:
# provisioning_scripts/terraform/common/localmachine/upload_scripts_to_there.bash
# provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/environment-box/local_manual__setup_at_creation.bash

# put in the right place.s
# for ssh and scp commands, respectively
export REMOTE_SSH_ADDR="$PAPERSPACE_USERNAME@$PAPERSPACE_IP"
export REMOTE_SCP_REF="$PAPERSPACE_USERNAME@$PAPERSPACE_IP"
export REMOTE_HOME_ABS_DIR="/home/$PAPERSPACE_USERNAME"

set -u ; echo "$REMOTE_HOME_ABS_DIR" > /dev/null  # assert env $REMOTE_HOME_ABS_DIR is set.

export SCRIPTS_BASE_REMOTE="$REMOTE_HOME_ABS_DIR/scripts-sosi"
export SCRIPTS2PUSH_DIR_REMOTE="$SCRIPTS_BASE_REMOTE/scripts_to_push"
# deprecate SCRIPTS_DIR_REMOTE
# TARGET_SCRIPTS_DIR="/home/paperspace/scripts_to_push/"
# not trivial at all
# TARGET_SCRIPTS_DIR="/home/paperspace/scripts-sosi/scripts_to_push"
# TARGET_SCRIPTS_DIR="$REMOTE_HOME_ABS_DIR/scripts-sosi/scripts_to_push"
# TARGET_SCRIPTS_DIR="$SCRIPTS_DIR_REMOTE"
# TARGET_SCRIPTS_DIR="$SCRIPTS_DIR_REMOTE/scripts_to_push"
# TARGET_SCRIPTS_DIR="$SCRIPTS_BASE_REMOTE/scripts_to_push"
set -u ; echo "$SCRIPTS2PUSH_DIR_REMOTE" > /dev/null  # assert env $SCRIPTS2PUSH_DIR_REMOTE is set.
set -u ; echo "$SCRIPTS2PUSH_DIR_LOCAL" > /dev/null  # assert env $SCRIPTS2PUSH_DIR_LOCAL is set.

# used in `upload_scripts`, hence, `SCRIPT1_MSAC_LOCAL` i.e. local_manual__setup_at_creation.bash


# ok it seems we need to do proper ...
# old name: command_via_ssh
function remote_command_via_ssh {

  # local command_arg_array=("${@}") # local args=("$@")

  ssh $SSH_CLI_OPTIONS \
    "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" \
    "${@}"
    #"$command"
}

# cancelled, not used
function scp_file {
  # ssh_cp or cp_via_ssh
  local FILE="${1}"
  local TARGET="${2}"

  ls -alth "$FILE"
  realpath "$FILE"

  scp $SSH_CLI_OPTIONS \
    "$FILE" \
    "$PAPERSPACE_USERNAME@$PAPERSPACE_IP":"$TARGET"

}

#- function ssh_go_into_shell

function upload_scripts_legacy {
    # Three types of scripts to upload:
    #1. by TF
    #2. by this script: inception_script_manual.bash
    #3. ??local_manual__setup_at_creation.bash: does the copy

    # SCRIPT_FILE_LOCAL2="docker_user_fix.bash"
    #
    # moved to: environment_boxes/neurotalk/scripts_to_push/inception_script_manual.bash
    #cat <<'EOF'  > "$SCRIPT3_ISM_FILE_LOCAL"
    # EOF
    # cp environment_boxes/neurotalk/scripts_to_push/inception_script_manual.bash "$SCRIPT3_ISM_FILE_LOCAL"

    # set -u
    # echo "$REMOTE_HOME_ABS_DIR" > /dev/null  # assert env $REMOTE_HOME_ABS_DIR is set.

    #    # is copied. Now executed!
    #    # probably shouwld be moved
    #    # SCRIPT3_ISM_FILE_LOCAL is not used anyomre!
    #    SCRIPT3_ISM_FILE_LOCAL="$REPO_ROOT/environment_boxes/neurotalk/scripts_to_push/inception_script_manual.bash"
    #    # but REMOTE_HOME_ABS_DIR is dinefined there in that file
    #    # SCRIPT_FILE_REMOTE_PREDICTED_NAME="$REMOTE_HOME_ABS_DIR/environment_boxes/neurotalk/scripts_to_push/inception_script_manual.bash"
    #    SCRIPT_FILE_REMOTE_PREDICTED_NAME -> SCRIPT3ISM_FILE_REMOTE_PREDICTED_NAME
    #    SCRIPT3ISM_FILE_REMOTE_PREDICTED_NAME="$REMOTE_HOME_ABS_DIR/scripts_to_push/inception_script_manual.bash"
    # PREDICTIED NAME ^

    NC='\033[0m' # No Color
    BLUE='\033[0;34m'
    RED='\033[0;31m'

    # local_manual__setup_at_creation.bash
    # inception_script_manual.bash


    # todo: rename variable: SCRIPT1 -> SCRIPT1_MSAC_LOCAL

    SCRIPT1_MSAC_LOCAL="$REPO_ROOT/provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/environment-box/local_manual__setup_at_creation.bash"
    # WRONG!!!! echo "# Do these in the remote machine: "
    # cat "$SCRIPT3_ISM_FILE_LOCAL" \
    # cat "$SCRIPT1_MSAC_LOCAL" \
    #  | { echo -e "$BLUE"; cat; echo -e "$NC"; }

    #SSH_CLI_OPTIONS=...  # was
    #or source ... .env  # to do?

    # move out of this upload_script? but copies there. So the name should be changed? But the name shows "when" it shoudl be run.
    # (confusion)
    # aha, rerverse what is in what!
    # THIS should have been called local_manual__setup_at_creation.bash
    #    and that, upload_scripts_to_there.bash

    # copy of scripts is done here: `local_manual__setup_at_creation.bash`` aka $SCRIPT1_MSAC_LOCAL
    # export SCRIPT3_ISM_FILE_LOCAL # no longer used!
    export -f scp_file
    export SSH_CLI_OPTIONS
    bash "$SCRIPT1_MSAC_LOCAL"
    # todo: move the actual script of copying of scripts to there
    # "this, calling a fucntion of that" ---> subcommand, in there !
    # but that , calling a function of this, is not!

    # also: at update, a bit on: at boot (but not as separwtre one that her ewere fer to it. it can itself have it there privately)
    # not thongs about lovally (updatig the local ssh ?)

    # go_ssh
    # remote_command_via_ssh "bash"

}


upload_scripts_legacy
# upload_scripts
# ssh_go_into_shell


# This si after upload, but are doign already things. (Note: tehis file name needs ot be changed)
remote_command_via_ssh "sudo apt update -y"
remote_command_via_ssh "sudo apt install tree"
remote_command_via_ssh "tree $SCRIPTS_BASE_REMOTE"

###########################
# Sending a message to the user
#
# Here, we are in the local-machine
# But above message, (when showed to used before ssh), is gonna run inside the remote ("provision"ed) machine.

# Every file has two names now:
#    environment_boxes/neurotalk/scripts_to_push/inception_script_manual.bash
#    environment_boxes/neurotalk/scripts_to_push/inception_script_manual.bash





# It should use "$REMOTE_HOME_ABS_DIR", but it's defined there.
# should that copy er decide about htese paths?
# since it's better htat they are calculated int he same plce!
# unlike the PP appraoch, where we do it in parallerl: two bands
# but we keep them synced!
#   but another mechnism: PP ! the PP loop! (refreshing cycle: and probably fast)
#   the quantificaiotn of the timescales of it:
#      the frequency of it!

# The ISM script
SCRIPT3_ISM_FILE_LOCAL="$SCRIPTS2PUSH_DIR_LOCAL/inception_script_manual.bash"
SCRIPT3ISM_FILE_REMOTE_PREDICTED_NAME="$SCRIPTS2PUSH_DIR_REMOTE/inception_script_manual.bash"
    # PREDICTIED NAME ^
#    but I see another name: my_scripts_put_here_via_scp

# (Just) base-names must match
grc diff <(basename "$SCRIPT3_ISM_FILE_LOCAL") <(basename "$SCRIPT3ISM_FILE_REMOTE_PREDICTED_NAME")



# We are in local machine, but below will appeat (as if ) on remote machine. We are at the edge (hence, this line needs to move up, just before the ssh command!)
echo 1>&2 -e "Upload done.
Now, it's
TIME to 🫱nually run this script on Remote Machine:
    🫱    bash $SCRIPT3ISM_FILE_REMOTE_PREDICTED_NAME
    "
