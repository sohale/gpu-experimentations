#!/bin/bash

# Adds C++ LSP support to your project
# Based on my previous solution setting it up: github.com/sohale/ifc2brep-0/scripts/lsp-server.bash
# https://github.com/sosi-org/playful_d3js/blob/d0a383e271c78c4777de0e91cf2f425ae731ac82/timeline-react/clean-up-and-run.bash


export PS4="🗣️  "
# SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
# timeline-react/clean-up.bash
INCLUDES="$(dirname "${BASH_SOURCE[0]}")/../includes"
echo "INCLUDES=$INCLUDES"
source $INCLUDES/gitrepo_root.bash
# source $INCLUDES/export_env.bash
# source $INCLUDES/export_func.bash

### MAIN ###
set -eux
gitrepo_root

# SETTINGS_FILE
export VSCODE_SETTINGSJSON="$REPO_ROOT/.vscode/settings.json"
ls -alth $VSCODE_SETTINGSJSON

export BACKUPS_DIR="$REPO_ROOT/.backups"
mkdir -p "$BACKUPS_DIR"

NEW_JSON_TAMPLATE='{
   "C_Cpp.intelliSenseEngine": "disabled",
   "clangd.path": "/usr/lib/llvm-18/bin/clangd",
   "clangd.arguments":
   [
      "-log=verbose",
      "-pretty",
      "--background-index",
      "--compile-commands-dir=/home/ephemssss/novorender/ifc2brep-0/scripts/"
   ],
   "placeholder": null
}'
# replace /usr/lib/llvm-18/bin

# MLIR_BIN="/mlir/llvm-project/build/bin"
LLVM_BIN_ON_HOST="/usr/lib/llvm-18/bin"
# adapted
REPLACEMENT_PATH="$LLVM_BIN_ON_HOST/"
# MODIFIED_JSON
ADAPTED_JSON=$(echo "$NEW_JSON_TAMPLATE" |sed 's|/usr/lib/llvm-18/bin/|'"$REPLACEMENT_PATH"'|g')

# jq -n '{}'

TEMP_JSONFILE=tmp_settings.json
jq ". + $ADAPTED_JSON"  "$VSCODE_SETTINGSJSON" > $TEMP_JSONFILE

# OUTPUTFILEN=$(mktemp)

echo "Make these changes:"

# diff <(jq . "$BACKUP_FILE") <(jq . "$TEMP_FILE")
grc diff $VSCODE_SETTINGSJSON $TEMP_JSONFILE \
  || :

echo "^ Now go about and make these changes manually. (Never automate this)"

# TIMESTAMP=$(date +"%Y%m%d%H%M%S")
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
export BFN=$BACKUPS_DIR/vcsode_settings_json_$TIMESTAMP.json
#todo:  make sure does not exist
cp $VSCODE_SETTINGSJSON $BFN
chmod -w $BFN

