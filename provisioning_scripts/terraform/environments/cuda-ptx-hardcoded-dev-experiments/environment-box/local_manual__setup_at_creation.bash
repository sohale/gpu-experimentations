set -eux
echo "v0.0.3 : manually-triggred (button/script)"
# repeated
# run directly? or by? provisioning_scripts/terraform/common/localmachine/upload_scripts_to_there.bash ?

# set -eu

# names:
# environment_boxes/neurotalk/setup_at_creation.bash
# environment_boxes/neurotalk/local_manual__setup_at_creation.bash
# also: at update, a bit on: at boot (but not as separwtre one that her ewere fer to it. it can itself have it there privately)

# compare: (merge?)
#    terraform/common/localmachine/upload_scripts_to_there.bash
#    environment_boxes/neurotalk/local_manual__setup_at_creation.bash

# idempotent (hence, for update too)

# NOW I CAN
# do any ssh mkdir -p etc

# see the function command_via_ssh

# for ssh and scp commands, respectively
REMOTE_SSH_ADDR="$PAPERSPACE_USERNAME@$PAPERSPACE_IP"
REMOTE_SCP_REF="$PAPERSPACE_USERNAME@$PAPERSPACE_IP"
REMOTE_HOME_ABS_DIR="/home/$PAPERSPACE_USERNAME"

grc diff <(echo "$REMOTE_HOME_ABS_DIR") <(echo "/home/paperspace")

# note: we are in the local machine, in this script


# From the point of view of the remote machine:
# SCRIPTS_PLACED_THERE='$HOME/scripts-sosi/'  # there 's 'HOME' (not "")
SCRIPTS_PLACED_THERE="$REMOTE_HOME_ABS_DIR/scripts-sosi"
# == REMOTE_SCRIPTS_FOLDER

# be careful: tjis is ther first ever comnand on the remote machine
# should not start with this
# it can break, etc
#...
# update:
# yes, a "yes" from stdio
ssh $SSH_CLI_OPTIONS "$REMOTE_SSH_ADDR" \
    "mkdir -p $SCRIPTS_PLACED_THERE/"
    # 'mkdir -p "$HOME/scripts-sosi/"'
    # todo: mkdir -p "$REMOTE_HOME_ABS_DIR/scripts-sosi/"'

ssh $SSH_CLI_OPTIONS "$REMOTE_SSH_ADDR" \
    "mkdir -p $REMOTE_HOME_ABS_DIR/secrets/"




#scp $SSH_CLI_OPTIONS \
#  "$SCRIPT_FILE" \
#  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP"/my_scripts_put_here_via_scp.bash

#    SCRIPT_FILE=.... --> scripts_to_push/inception_script_manual.bash
set -x
# skipping: Now as part of "scp -r" command below.
: || \
scp_file "$SCRIPT_FILE" "/home/paperspace/my_scripts_put_here_via_scp.bash"
# SCRIPT_FILE is: inception_script_manual.bash"
# note: SKIPPED ^

# todo:
# scp_file "$SCRIPT_FILE" "/home/paperspace/my_scripts_put_here_via_scp.bash"

# ls environment_boxes/neurotalk/scripts_to_push/system_hardware_spec_info.bash


HERE_LOCAL="$REPO_ROOT"
# LOCAL_SCRIPTS_FOLDER
# LOCAL_HERE_SCRIPTS_FOLDER="$HERE_LOCAL/environment_boxes/neurotalk/scripts_to_push"
LOCAL_HERE_SCRIPTS_FOLDER="$HERE_LOCAL/environment_box/scripts_to_push"

# alternativd naming: LOCAL_SCRIPTS_FOLDER, REMOTE_SCRIPTS_FOLDER

# scp_file
scp \
    -r \
    $SSH_CLI_OPTIONS \
    "$LOCAL_HERE_SCRIPTS_FOLDER/" \
    "$PAPERSPACE_USERNAME@$PAPERSPACE_IP":"$SCRIPTS_PLACED_THERE/"

# rsync -avz --delete "$LOCAL_HERE_SCRIPTS_FOLDER/"  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP":"$SCRIPTS_PLACED_THERE/"
echo "recursive scp done."

# secret
# Now, instead, copied by TF!
# Keep separate from the one by TF
scp \
    $SSH_CLI_OPTIONS \
    "$HERE_LOCAL/demo_neurotalk/secrets/ghcli-token.txt" \
    "$PAPERSPACE_USERNAME@$PAPERSPACE_IP":"$REMOTE_HOME_ABS_DIR/secrets/ghcli-token-1.txt"
