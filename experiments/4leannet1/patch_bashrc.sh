#!/bin/bash

# names: bash_patch_history.bash patch_bashrc.sh

# Path to the .bashrc file
BASHRC="$HOME/.bashrc"

# Your desired changes as functions

append_if_not_exist() {
    local line="$1"
    local file="$2"
    grep -qxF -- "$line" "$file" || echo "$line" >> "$file"
}

# Example changes
append_if_not_exist 'export HISTCONTROL=ignoredups:erasedups' "$BASHRC"
append_if_not_exist 'shopt -s histappend' "$BASHRC"
append_if_not_exist 'PROMPT_COMMAND="history -a;$PROMPT_COMMAND"' "$BASHRC"
append_if_not_exist 'export HISTSIZE=10000' "$BASHRC"
append_if_not_exist 'export HISTFILESIZE=20000' "$BASHRC"

# Reload .bashrc (optional, for the current session)
source "$BASHRC"


