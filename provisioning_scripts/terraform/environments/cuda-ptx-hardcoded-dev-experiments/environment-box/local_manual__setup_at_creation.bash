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



ssh $SSH_CLI_OPTIONS "$REMOTE_SSH_ADDR" \
    "mkdir -p $REMOTE_HOME_ABS_DIR/secrets/"




#scp $SSH_CLI_OPTIONS \
#  "$SCRIPT_FILE" \
#  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP"/my_scripts_put_here_via_scp.bash

#    SCRIPT_FILE=.... --> scripts_to_push/inception_script_manual.bash
set -x
# skipping: Now as part of "scp -r" command below.
: || \
scp_file "$SCRIPT_FILE" "$REMOTE_HOME_ABS_DIR/my_scripts_put_here_via_scp.bash"
# SCRIPT_FILE is: inception_script_manual.bash"
# note: SKIPPED ^

# todo:
# scp_file "$SCRIPT_FILE" "REMOTE_HOME_ABS_DIR/my_scripts_put_here_via_scp.bash"

# ls environment_boxes/neurotalk/scripts_to_push/system_hardware_spec_info.bash

# SCRIPTS_DIR_LOCAL:
# HERE_LOCAL="$REPO_ROOT"
# LOCAL_SCRIPTS_FOLDER
# LOCAL_HERE_SCRIPTS_FOLDER="$HERE_LOCAL/environment_boxes/neurotalk/scripts_to_push"
# LOCAL_HERE_SCRIPTS_FOLDER="$TF_BASEDIR/environment-box/scripts_to_push"
# LOCAL_SCRIPTS_FOLDER -> LOCAL_HERE_SCRIPTS_FOLDER -> SCRIPTS_FOLDER_LOCAL -> SCRIPTS_DIR_LOCAL
# LOCAL_HERE_SCRIPTS_FOLDER, SCRIPTS_FOLDER
# SCRIPTS_DIR_LOCAL

# From the point of view of the remote machine:
# REMOTE_SCRIPTS_FOLDER , SCRIPTS_PLACED_THERE -> SCRIPTS_FOLDER_REMOTE -> SCRIPTS_DIR_REMOTE
# SCRIPTS_DIR_REMOTE='$HOME/scripts-sosi/'  # there 's 'HOME' (not "")
# export SCRIPTS_DIR_REMOTE="$REMOTE_HOME_ABS_DIR/scripts-sosi"
export SCRIPTS_BASE_REMOTE="$REMOTE_HOME_ABS_DIR/scripts-sosi"
export SCRIPTS2PUSH_DIR_REMOTE="$SCRIPTS_BASE_REMOTE/scripts_to_push"


# alternative naming: LOCAL_SCRIPTS_FOLDER, SCRIPTS_DIR_REMOTE
# alternative naming: SCRIPTS_DIR_LOCAL, SCRIPTS_DIR_REMOTE
# alternative naming: .., SCRIPTS_BASE_REMOTE
# alternative naming: SCRIPTS2PUSH_DIR_LOCAL, SCRIPTS2PUSH_DIR_REMOTE

set -u
echo "$SCRIPTS2PUSH_DIR_LOCAL" > /dev/null
test -d "$SCRIPTS2PUSH_DIR_LOCAL"


# be careful: this is the first ever comnand on the remote machine
# should not start with this
# it can break, etc
#...
# update:
# yes, a "yes" from stdio
ssh $SSH_CLI_OPTIONS "$REMOTE_SSH_ADDR" \
    "mkdir -p $SCRIPTS_BASE_REMOTE/"

#########################
# Bulk-copy the scripts:
#########################
# scp_file
scp \
    -r \
    $SSH_CLI_OPTIONS \
    "$SCRIPTS2PUSH_DIR_LOCAL/" \
    "$PAPERSPACE_USERNAME@$PAPERSPACE_IP":"$SCRIPTS2PUSH_DIR_REMOTE/"

# why "delete" ??
# rsync -avz --delete "$SCRIPTS2PUSH_DIR_LOCAL/"  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP":"$SCRIPTS2PUSH_DIR_REMOTE/"
echo "recursive scp done."

#########################
# Copy the secret
#########################
# secret
# Now, instead, copied by TF!
# Keep separate from the one by TF
scp \
    $SSH_CLI_OPTIONS \
    "$EXPERIMENT_TFVARS/ghcli-token.txt" \
    "$PAPERSPACE_USERNAME@$PAPERSPACE_IP":"$REMOTE_HOME_ABS_DIR/secrets/ghcli-token-1.txt"

# Compromise: $EXPERIMENT_TFVARS is used for non-TF secret too.
