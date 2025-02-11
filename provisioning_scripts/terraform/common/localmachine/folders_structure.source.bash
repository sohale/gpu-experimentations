# folder names shared across all commands related to terraform here.
# Always use things defines here using `source` command first.
# Can have Local Machine, or Remote folders.
# This file alone represents the organisaiton of everthing including:
#   * local terraform runtime
#   * local terraform scripts (.tf code), config (tfvars), etc
#   * usage forder's structure if any terraform-related files are stored there.

# Symbolically (H/U), which means, informs about remoote machine folders too (if I success in this pattern)

export REPO_ROOT="$HOME/gpu-experimentations"

# A half-baked idea: some envs shold be generatpr-like, that can be used in the script of interest, not in this common file, but, are "code".
#   It won't help the principal/prime purpose of this: because, the prime purpose is to have a common place for all locaitons almos tdeclaratively in one place.
# function GEN__this_script_dir {
#    _THIS_SCRIP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# }


# On: local machine or workstation
export EXPERIMENT_DIR="$REPO_ROOT/experiments/11_matrix_cuda"
# old: EXPERIMENT_DIR="$REPO_ROOT/demo_neurotalk"
# export EXPERIMENT_TFDIR="$REPO_ROOT/provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments"
export TF_BASEDIR="$REPO_ROOT/provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments"
export RUNTIME="$TF_BASEDIR/runtime"


# export EXPERIMENT_TFVARS="$EXPERIMENT_DIR/tfvarconfig"
export EXPERIMENT_TFVARS="$TF_BASEDIR/tfvarconfig"
# export EXPERIMENT_TFVARS="$TF_BASEDIR/config"


# TF_MAIN_TF_DIR and TF_BASEDIR are the same, here in this new organisation.
# TF_MAIN_TF_DIR: is : # The CWD of the terraform command:
# The CWD of the terraform command: (where the main.tf is): = TF_MAIN_TF_DIR = TF_MAIN_CWD
TF_MAIN_TF_DIR="$TF_BASEDIR"
test -f "$TF_MAIN_TF_DIR/main.tf"


TF_STATE_FOLDER="$RUNTIME/terraform_state"
TF_STATE_FILE="$TF_STATE_FOLDER/terraform.tfstate"
TF_DIFF_OUT_FILE="$TF_STATE_FOLDER/plan_delta.dfplan"

# SCRIPTS_DIR_LOCAL
export SCRIPTS2PUSH_DIR_LOCAL="$TF_BASEDIR/environment-box/scripts_to_push"
