set -eu
# set -x
set -x

# previous name: up.bash
# previous name: up-matmul-cuda-experiment.bash


echo 'usage:
bash provisioning_scripts/terraform/common/localmachine/up-matmul-cuda-experiment.bash [subcommand]
example:
bash provisioning_scripts/terraform/common/localmachine/up-matmul-cuda-experiment.bash tfinit
'
# Useful:
# TF_LOG=DEBUG

export REPO_ROOT="$HOME/gpu-experimentations"
# source ~/gpu-experimentations/provisioning_scripts/terraform/common/localmachine/folders_structure.source.bash
source "$REPO_ROOT/provisioning_scripts/terraform/common/localmachine/folders_structure.source.bash"
# export REPO_ROOT="$HOME/gpu-experimentations"
# Uses:
echo "$REPO_ROOT $EXPERIMENT_DIR $TF_BASEDIR $RUNTIME $EXPERIMENT_TFVARS $TF_MAIN_TF_DIR"> /dev/null

_THIS_SCRIP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FROMLOCAL_SCRUPTS="$_THIS_SCRIP_DIR"

# terraform plan ... -target=resource

mkdir -p "$EXPERIMENT_TFVARS"

test -f "$TF_MAIN_TF_DIR/main.tf"  # "Error: Terraform configuration files not found in the specified directory ($TF_MAIN_TF_DIR)."


mkdir -p "$TF_STATE_FOLDER"

# was: "tfconfig.tfvars"
TF_PROJ_VARFILE="$EXPERIMENT_TFVARS/current_active.tfvars"
TF_SECRETS_VARFILE="$EXPERIMENT_TFVARS/current_active_secrets.tfvars"

# streategy: fail-fast, but, give actionable message: (even the command)
test -L "$TF_PROJ_VARFILE" || {
    # todo, may move. Undecided.
    echo "For you to do:"
    echo 'pushd $(pwd);'" cd $EXPERIMENT_TFVARS; ln -s  entomind.tfvars  current_active.tfvars ; popd"
    exit 1
}
test -L "$TF_SECRETS_VARFILE" || {
    echo "For you to do:"
    echo 'pushd $(pwd);'" cd $EXPERIMENT_TFVARS; ln -s  secrets_entomaind.tfvars  current_active_secrets.tfvars; popd"
    exit 1
}


# Shall/should be symbolic link. These can (& should) be a symlink:   (to, e.g. secrets_sosi.tfvars )
test -L "$TF_PROJ_VARFILE"
test -L "$TF_SECRETS_VARFILE"
# Alternatively: provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/config/

test -f "$TF_MAIN_TF_DIR/main.tf"
test -f "$TF_SECRETS_VARFILE"
test -f "$TF_PROJ_VARFILE"
test -d "$EXPERIMENT_TFVARS"



# how to read: "subcommand prefix"
_scprefix="________subcommand___"

function ________subcommand___tfinit {
# Does  TF_STATE_FILE override  terraform/environments/neurotalk/backend.tf ?

cd "$TF_MAIN_TF_DIR" && pwd
# terraform init -var-file="secrets.tfvars"  -var-file="tfconfig.tfvars"
terraform init \
    -var-file="$TF_PROJ_VARFILE" \
    -var-file="$TF_SECRETS_VARFILE" \
    -backend-config="$TF_STATE_FILE" \
     || :

# How to fix this?
# "backend-config was used without a "backend" block in the configuration"
# is the "-backend-config=" to "override the default local backend configuration"?
#   What does it mean?

# This `-backend-config=` perhaps implied the init should be expected to be already there?
#    -backend-config="$TF_STATE_FILE" \
# for init only, not plan
# "init" does not read the .tfvars (tfconfig.tfvars)
}

# No such option:     -chdir="$TF_MAIN_TF_DIR" \

function ________subcommand___tfplan {

cd "$TF_MAIN_TF_DIR" && pwd

terraform plan \
    -var-file="$TF_PROJ_VARFILE" \
    -var-file="$TF_SECRETS_VARFILE" \
    -out "$TF_DIFF_OUT_FILE" \
    # || :
# "plan" has no    -backend-config

  #  Why `-out` arg, I mean the "$TF_DIFF_OUT_FILE", is only in the `plan`, but why it is no INPUT of anything?!
}

function env_exporter {

    # Extract the outputs from Terraform-land, into the bash / env-land
    function pretend_to_assign_code {
      # pretend to code that assisgned
      # preted code that assigns
      # code that pretends to assign
      # pretend_to_assign
      # wrong old name: pipe_to_env

      # 74.82.28.237=${74.82.28.237}
      # xargs -n1 | awk '{print $1"=${"$1"}"}'
      xargs -n1 | awk '{print "export "$1"=\"${"$1"}\""}'
    }

    # env
    # printenv is a wrong approach
    # printenv PAPERSPACE_IP PAPERSPACE_USERNAME
    # printenv PAPERSPACE_IP PAPERSPACE_USERNAME | sed 's/^/export /' \
    # printenv PAPERSPACE_IP PAPERSPACE_USERNAME | pipe_to_env \
    #  > "$PIPE_TEMP_FILENM"  # > $PIPE1
    #  | `create machine_.v1.env.buffer`
    # echo "PAPERSPACE_IP PAPERSPACE_USERNAME" | xargs -n1 | awk '{print $1"=${"$1"}"}' | envsubst \
    #  | machine_.v2.env.buffer
    # echo "PAPERSPACE_IP PAPERSPACE_USERNAME" | pretend_to_assign_code | envsubst
    #    > "$PIPE_TEMP_FILENM"  # > $PIPE1

    # the expoerter:
    pretend_to_assign_code | envsubst
}

function tell_me_about_file {
    # "tell me about it"
    # tells me about it (a file). Instead of realpath:
    # realpath $1
    # output:
    # fd1 (stdout) :  the normalised filename
    # fd2 (stderr):   shows on terminal

    local FNAME="${1}"

    REPO_ROOT="/myvol/pocs_for_nikolai"
    echo 1>&2 "$FNAME  Full path: $( realpath  --relative-to="$REPO_ROOT" $FNAME )"
    cat 1>&2 "$FNAME"

    # The main (fg) output to stdout:
    # RELPATH="$(
    realpath  --relative-to="$REPO_ROOT" $FNAME
    # )"
}

function _capture_outputs {
    # Inputs: a bunch of 'terraform output' commands.

    #PIPE1=$(mktemp -u)
    #PIPE2=$(mktemp -u)
    #mkfifo $PIPE1 $PIPE2
    # Don't use pipes, because the you iwll need to use `&`, which then you cannot use for env variables: because uyuo need to wait for it.
    # you need to wait inside a "$( cat <$PIPE1 )"

    # todo: change if necessary. not revised
    # TEMP_FOLDER="$REPO_ROOT/experiments/11_matrix_cuda/runtime/tf-temp-runtime"
    TEMP_FOLDER="$RUNTIME/tf-temp-runtime"
    mkdir -p "$TEMP_FOLDER"
    # This seems to be the only place this is used?

    test -d $TEMP_FOLDER
    PIPE_TEMP_FILENM="$(mktemp $TEMP_FOLDER/mmmXXXXXX -u)"

    export PAPERSPACE_IP="$(terraform output -raw public_ip_outcome)"
    export PAPERSPACE_USERNAME="$(terraform output -raw username_outcome)"

    echo "PAPERSPACE_IP PAPERSPACE_USERNAME" | \
    env_exporter  \
    > "$PIPE_TEMP_FILENM"  # > $PIPE1
    # echo 1>&2 "$PIPE_TEMP_FILENM"
    # cat  1>&2 "$PIPE_TEMP_FILENM"

    # instance_id

    # lazy
    #luxargs --dry-run -- ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$PAPERSPACE_IP"
    #luxargs --dry-run -- ssh -v -i ~/.ssh/paperspace_sosi_fromlinux $PAPERSPACE_USERNAME@$PAPERSPACE_IP

    # save buffer1 as a file
    local machine_tuple_used_as_string="$(
      terraform output -json machine_tuple_used | jq -r '. | join("-")'
    )"
    # dodo: this ^ can cause error ^: Output "machine_tuple_used" not found

    echo 1>&2 "machine_tuple_used_as_string=$machine_tuple_used_as_string"
    # cat < `use machine_.v2.env.buffer` > machine_.v2.env.buffer
    # cat <$PIPE1 > "machine_${machine_tuple_used_as_string}.env"
    FNAMEB="machine_${machine_tuple_used_as_string}.env"
    FNAME="$TEMP_FOLDER/$FNAMEB"
    cp "$PIPE_TEMP_FILENM"  $FNAME

    # tell_me_about_file "$FNAME"

    echo 1>&2 -n "The .env file saved: "

    RELPATH="$(tell_me_about_file "$FNAME")"
    echo 1>&2 "
    source \"$RELPATH\""'
    ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$PAPERSPACE_IP"
    ssh -v -i ~/.ssh/paperspace_sosi_fromlinux $PAPERSPACE_USERNAME@$PAPERSPACE_IP
    '

    # this is upstream, so it shall not take it (paperspace_sosi_fromlinux) from tf.

    # Unfortunately, we dont have SSH_CLI_OPTIONS here
    # echo "${SSH_CLI_OPTIONS:-wha1}" || { echo oops ; }

    # Clean up
    # rm $PIPE1 $PIPE2
    rm "$PIPE_TEMP_FILENM"
}


# refresh
# other comands
function ________subcommand___other {
    shift 2
    echo "gonna run:"
    luxargs : "$@"

    # not used yet
    function _print_args_ {
        local args=("$@")
        for arg in "${args[@]}"; do
            echo "[$arg]"
        done
    }
    _print_args_ "$@"


    cd "$TF_MAIN_TF_DIR" && pwd
    echo -e "\n\n\n"
    luxargs "$@"
}

# moved:
# function scp_file { ...}
# ssh_go_into_shell
# command_via_ssh

function ________subcommand___show_outputs {

    # Oh, no, it does a lot more:
    #     capture outputs
    #     refresh
    #     start the machine?!
    #     copies scripts (for manual execution)
    #     extracts machnie name? (machine id?)
    #     ...
    #    goes into ssh?

    cd "$TF_MAIN_TF_DIR" && pwd
    echo -e "\n\n\n"

    # almost apply, but refreshes in case I have modified the outputs.tf between the last tfapply and this
    # runs the pipe of "apply" -> "collect outputs"
    #   in fact, outputs aee only ... as if we do a "terraform refresh" (whouisl be somehow called teraform popluate output or simply "terraform output" but that is given another meaning. note: that reverse-grammar ... is here )
    #   no: tf refresh, actully seeds it form the remote "resource" (not the local state file)
    : || \
    terraform refresh \
        -var-file="$TF_PROJ_VARFILE" \
        -var-file="$TF_SECRETS_VARFILE"

    _capture_outputs

    local machine_id="$(
      terraform output -raw instance_id
    )"

    echo 1>&2 "show_outputs: done _capture_outputs"


    echo 1>&2 "show_outputs: going to:  turn on the compueter & refresh the ssh 'known_hosts' "

    # Do this before the upload_scripts
    ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$PAPERSPACE_IP"

    echo 1>&2 "show_outputs: going to:  pspace"

    # this starts too?!
    {
      echo
      pspace machine list
      echo
      pspace machine start "$machine_id"
    } || {  echo \
      "The 'pspace' command failed or not available, or machine already start(ed/ing)" ;
    }

    echo 1>&2 "show_outputs: going to:  upload_scripts"


    # SSH_CLI_OPTIONS="-v -i $SSH_KEY_PATH"
    SSH_CLI_OPTIONS='-v -i ~/.ssh/paperspace_sosi_fromlinux'
    SSH_CLI_OPTIONS='-i ~/.ssh/paperspace_sosi_fromlinux'
    echo "ssh $SSH_CLI_OPTIONS  '$PAPERSPACE_USERNAME@$PAPERSPACE_IP'"
    # source ...

   echo "may turn off, prevent subsequeent confirmations, etc"
    # aab  bandi
    # or as "after ssh-keygen"
    # echo -ne "yes\n\n\n" | \  # did onot work
    ssh $SSH_CLI_OPTIONS  \
        -o StrictHostKeyChecking=no  \
        "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" "sudo touch /i-was-here"


# This option controls the behavior of SSH when connecting to a new host for the first time or if the host key has changed.
# Disabling StrictHostKeyChecking and using /dev/null for UserKnownHostsFile means that you are not verifying the host's identity,
# (otherwise) prevents "man-in-the-middle" attacks.


    export SSH_CLI_OPTIONS
    # export -f scp_file
    # upload_scripts
    bash "$FROMLOCAL_SCRUPTS/upload_scripts_to_there.bash"


    function ssh_go_into_shell {
        ssh $SSH_CLI_OPTIONS  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP"
    }

    ssh_go_into_shell

    echo 1>&2 "show_outputs: finished the happy path."

}

function ________subcommand___bash {

    echo "Runs an interactive subshell in the remove machine UP-ed by terraform"

    cd "$TF_MAIN_TF_DIR" && pwd
    echo -e "\n\n\n Your interactive subshell:"
    # PATH="$PATH"


    #bash   -c '
    bash -c "$(cat <<'EOF_STARTUP'
        pwd;
        export PROMPT_COMMAND='{ __exit_code=$?; if [[ $__exit_code -ne 0 ]]; then _ps1_my_error="🔴${__exit_code}"; else _ps1_my_error=""; fi; }';
        # PS1="PROMPT: \" \w \"  \[\033[00;31m\]${_ps1_my_error}\[\033[01;32m\] >>>> \n> " exec bash --norc
        export PS1='\[\033[01;33m\]𝓜𝓛𝓘𝓡 \[\033[00;34m\]container:@\h \[\033[01;34m\]\w\[\033[00m\]\n\[\033[01;32m\]$(whoami)\[\033[00m\]  \[\033[00;31m\]${_ps1_my_error}\[\033[01;32m\] \$ \[\033[00m\]'
EOF_STARTUP
)"


}



function ________subcommand___tfapply {

    # does not the machine? or does it?
    # need to run `show_output` too?

    cd "$TF_MAIN_TF_DIR" && pwd

    terraform apply \
        -var-file="$TF_PROJ_VARFILE" \
        -var-file="$TF_SECRETS_VARFILE" \
        # -out "$TF_DIFF_OUT_FILE" \

    return 0
    # Done. Now capture the output
    # _capture_outputs
    {
      _capture_outputs \
        || { _e="$?" ; echo "error with exit code: $?"; exit _e; }
    } &
    echo 1>&2 "Back to tfapply: waiting."
    wait

    echo 1>&2 "tfapply finished the happy path."
}

function ________subcommand___tfdestroy {
    cd "$TF_MAIN_TF_DIR" && pwd
    terraform destroy \
        -var-file="$TF_PROJ_VARFILE" \
        -var-file="$TF_SECRETS_VARFILE" \

}

# Do we need a tfvalidate ?
function ________subcommand___tfvalidate {
    cd "$TF_MAIN_TF_DIR" && pwd

    terraform validate

    #  -var-file="$TF_PROJ_VARFILE" \
    #  -var-file="$TF_SECRETS_VARFILE" \
    : || '
    The terraform validate command validates the configuration files in a directory,
    referring only to the configuration and
    not accessing any remote services such as remote state, provider APIs, etc.
    Whether a configuration is syntactically valid
    and internally consistent,
    regardless of any provided "variables" or existing state.
    '
}

: || {
function ________subcommand___ssh_into {
    source "$RELPATH"
    ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$PAPERSPACE_IP"

    echo "ohno1"
    echo "${SSH_CLI_OPTIONS:-ohno2}" || { echo oops; }

    ssh \
      -v \
      -i ~/.ssh/paperspace_sosi_fromlinux $PAPERSPACE_USERNAME@$PAPERSPACE_IP
}
 }

# Helpers for usage message & reflections (list of subcommands)
function subcommands_list {
    # list of subcommands:
    declare -F | grep -E "^declare -f $_scprefix" | awk '{print $3}' | sort \
      | sed "s/$_scprefix//g" \

    #  | sed "s/$_scprefix/ 🐸  /g"
}
function subcommands_usage {
  echo "subcommand can be either { $(subcommands_list | paste -s --delimiters="," - ) } : use one of the below:"

  # | sed 's/^/        ${0} /'
  subcommands_list \
    | xargs -I{} \
      echo \
"        ${0}   "'{}'
}

# Execute the sub-command !
sub_command=${1:-""}

_luxargs="$(which luxargs || echo -n "" )"
_luxargs="${_luxargs:-""}"


if $_luxargs [ -z "${sub_command}" ]; then

  subcommands_usage

  exit 1
fi

function escapeq_me {
  echo "\"$(printf "%q" "${1}")\""
}

subcommands_usage | grep "${sub_command}" || {
  echo "Not found: $(escapeq_me "${sub_command}")"
  subcommands_usage
  exit 1
}

function_to_call="${_scprefix}${sub_command}"

# run ! the sub command
"$function_to_call"

echo 1>&2 'good'

# [1] Also see ./scripts/neurotalk
# [2] Based on: https://github.com/sohale/gpu-experimentations/blob/main/experiments/raw1/localmachine-tfinit.bash
# [3] For instalation, see: terraform/common/localmachine/install-terraform-on-localmachine.bash
# [4] https://blog.paperspace.com/introducing-paperspace-terraform-provider
# First error from `tf init``: 'The directory has no Terraform configuration files. You may begin working with Terraform immediately by creating Terraform configuration files.
# [5] ...
