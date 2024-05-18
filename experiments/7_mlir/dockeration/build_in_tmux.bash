#!/bin/bash
# set -e  # causes issue for tmux
set -xu

# SESSION_NAME="mlir-dev-docker-build-tmuxsession"
# LOG_FILE="build1.stdout.log"
# LOGERR_FILE="build1.stderr.log"
# SESSION_PREFIX="mlirdockerbuild"
SESSION_PREFIX="mlirdbuild"
SESSION_TIMESTAMP="$(date +'d%Y%m%d_t%H%M%S')"  # time of initiation
SESSION_NAME="${SESSION_PREFIX}_${SESSION_TIMESTAMP}_tmuxsession"
LOG_FILE="./tmux_logs/${SESSION_NAME}.stdout.log"
LOGERR_FILE="./tmux_logs/${SESSION_NAME}.stderr.log"
mkdir -p ./tmux_logs

DOCKER_BUILD_SCRIPT="./build-mlir-docker.bash"
# for testing this script:
# DOCKER_BUILD_SCRIPT="ls -alth; echo "ERR" &1>&2; sleep 2;ls -alth; sleep 2"


# Start a tmux server if not running
# tmux start-server # || :
tmux -V
# tmux ls   2> /dev/null  ||  :
tmux ls || :

# For convenience:
cat << TMUX.CONF  > .local_mlirdb_tmux.conf
# For setup:
#   MacBook M1 : iTerm2 → ssh → buntu 22.04.4 LTS → tmux 3.2a

# Enable mouse support
set-option -g mouse on

# Increase scrollback buffer size to 100,000 lines (for a new tmux session only)
set -g history-limit 100000

# Bind mouse wheel to scroll in copy mode
bind -T copy-mode-vi WheelUpPane send-keys -X scroll-up
bind -T copy-mode-vi WheelDownPane send-keys -X scroll-down
TMUX.CONF

tmux source-file .local_mlirdb_tmux.conf

# Check if tmux session already exists
tmux has-session -t $SESSION_NAME  # 2>/dev/null

if [ $? != 0 ]; then
    # Create a new tmux session and run the command
    tmux new-session -d  -s $SESSION_NAME
    # tmux send-keys -t $SESSION_NAME  "$DOCKER_BUILD_SCRIPT    $LOG_FILE  2> $LOGERR_FILE" C-m
    tmux send-keys -t $SESSION_NAME  "{ $DOCKER_BUILD_SCRIPT; }  1> >(tee -a $LOG_FILE) 2> >(tee -a $LOGERR_FILE >&2)" C-m
else
    # Attach to the existing session and run the command
    tmux send-keys -t $SESSION_NAME  "{ $DOCKER_BUILD_SCRIPT; }  1> >(tee -a $LOG_FILE) 2> >(tee -a $LOGERR_FILE >&2)" C-m
    :
fi

ls -1 ~/.bash_history 1>/dev/null  # to check if we use bash
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export NC='\033[0m'  # Color Reset

cmd1="tmux attach-session -t $SESSION_NAME"
echo -e "${GREEN}$cmd1${NC}"
echo "^ Run to attach to the tmux build session, if you want to see the output live, run:"
echo "Control+B, then: ? for help, D to detach, q for back"
# echo "$cmd1" >> ~/.bash_history
# exec bash  # apply the new history entry
