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
# moved back here? not sure:


function append_remote_bashrc {
    # dynamically genrated file
    PIPE_TEMP_FILEN="$(mktemp $TEMP_FOLDER/mmmXXXXXX -u)"
    cat > $PIPE_TEMP_FILEN <<'EOF_STARTUP'
        # This is the content of ~/scripts-sosi/dynamically-generated-replaced.source.bash
        # NOT  my -tf-s ct ip ts-ar e - ap pended
        cat ~/.bashrc
        echo "This is called by .bashrc:  \$\$=$$"
        pwd;

        # Even more complicated. todo: remove this from other scripts?:

        # alternatives:
        # compare: (probably local vs remote?)
        #   on: local machine
        # provisioning_scripts/terraform/common/per-provider/paperspace/setup_anew_ssh.bash
        #    on: remote machine, ceation time. (ad script name! the reason, it creates an env file)
        # provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/environment-box/scripts_to_push/ghcli-login.bash
        #   after that, there are two ways: "source the env file", or "directly eval and ssh-add", manually (manually only?)

        # not scriptable?
        # eval "$(ssh-agent -s)"
        # ssh-add /home/paperspace/.ssh/github_sosi_from_paperspace

        # Let me do it
        # no, parametrising using env is hard:
        #export REMOTE_HOME_ABS_DIR="/home/$PAPERSPACE_USERNAME"
        #export SCRIPTS_BASE_REMOTE="$REMOTE_HOME_ABS_DIR/scripts-sosi"
        #SCRIPTS_BASE_REMOTE=
        #source $SCRIPTS_BASE_REMOTE/refresh_ssh_agent.env

        # Although covered in that static file: dot_bashrc.bash (Is π4)
        source ~/scripts-sosi/refresh_ssh_agent.env
        {
        gh --version
        gh auth status
        } || :


        # Although covered in that static file: dot_bashrc.bash  (Is π4)
        sudo timedatectl set-timezone UTC

        # (π5)
        # export PS1="PROMPT: \" \w \"  \[\033[00;31m\]${_ps1_my_error}\[\033[01;32m\] >>>> \n> " exec bash --norc
        export PROMPT_COMMAND='{ __exit_code=$?; if [[ $__exit_code -ne 0 ]]; then _ps1_my_error="${__exit_code} 🔴"; else _ps1_my_error=""; fi; }';
        export PS1='\[\033[01;33m\]➫ 𝗚𝗣𝗨 \[\033[00;34m\]container:@\h \[\033[01;34m\]\w\[\033[00m\]\n➫ \[\033[01;32m\]$(whoami)\[\033[00m\]  \[\033[00;31m\]${_ps1_my_error}\[\033[01;32m\] \$ \[\033[00m\]'
        echo -en '𝗚𝗣𝗨\n𝙶𝙿𝚄\n𝔊𝔓𝔘\n𝑮𝑷𝑼\n𝓖𝓟𝓤\n𝐆𝐏𝐔\n𝕲𝕻𝖀\nＧＰＵ\n🄶🄿🅄\n𝒢𝒫𝒰\n\n'
EOF_STARTUP

    : "
    Note on above ^ script:
        Since this has prompt, and the π2 is covered by `dot_bashrc.bash`, this is perhaps shell/interactive--level (π3)
        So, perhaps, this is actually not necessary.
        The (execution-threadchain)-map is:
            dynamically-generated-replaced.source.bash
            generated at time of "show_outputs" (deployment, BTW, what is the difference beween "show_outputs" and main.tf (tfapply)'s own script(s)? )
        Need a way to characerise: chain: script that triggers/starts [interactive,etc] (not itself), --> the bash (shell interactive sessions): itself, --> calls .bashrc, which calls this, and, the static one

        Three features characetrise / specify (each) script/invokation:
        * static-ness (veriosn control) vs dynamic-ness (generated: at runtime (but deploy) )
        * The πi-ness
        * The execution "chain" (thread/chain)

        not imporant: in the (inline "bash -c"), or not, or via its --bashrc, etc
        All this analysis (and below πi), leads me to this: in the rsync2, use (`source`) dot_bashrc.bash


    The `dynamically-generated-replaced.source.bash` happens to be the one for prompt (gas): ( interactive : π4 =  π2)

    Note:
        π1. permamnet (changes, like the change of .bashrc itself: once, and remains)
        π2. per login-session: for future (or following) scripts: like github, timezone, sshagent (for all to-remote [sub]commands, not just interactive ones)
        π3. per login-session, but shell/interactinve. like prompt
        π4. (in script, or before interactive: liquid) # why not π2?
        π5. gas: in interactive one, by user  # why not π3?

        ok, this helps me find out  hout where I should call ... (not in the script I am doing now in a subcommand_bash or subcommand_rync2 one, or even more surfaced: asking the user to type in)

        Interesrintgly, "π2" are covered in `dot_bashrc.bash`


        * Per-(turn-on) (restart, or, on), happsn to be the same as "per login-session"
        * The per login-session happens to be the same as per-login. (coz that's Unix/Linux)

        Theoretical "at" s:
        * Per-turn-on
        * Per-restart
        * Per-un-standby (wake)
        * Per-login (The env '$HOME' starts to be meaningful)
        * Per-session (same as use login, in Unix/Linux, but couls be a remote-desctop sesssion + a cmd.exe in windows)
        * Per shell script: prefix (not leading to interactive) (in theory, we can put a \"bash\" in the end, but it's not recommended to do it in this intention-path)
        * Per shell script, prefix, leading to interactive.
        * Per shell script, after the prefix: the interactive (e.g. setting the PS1)
        * Per shell script, after the prefix: entered by user, interactively: e.g. '~', 'eval ...'


        Which one does this set: `sudo timedatectl set-timezone UTC`
           i.e., Permanence: How deep the reset(!), this can .
           Markov-reset!
           "The permanenece orbit". Reset-depth! (Layers?! like Docker)

    Other details:
        How to formalise this?:
        refresh_ssh_agent.env is creted by ghcli-login.bash
            Also:
            Who runs ghcli-login.bash? It was missing. Formalise that too. It is hard to infer it (even runtime, but we want compile-time, static-analaysis-time)
            The question of missing `refresh_ssh_agent.env` is deferred to the questinn of involing `ghcli-login.bash`.
            ( and it is not the rong path. It is just not created: ots creation "code" is not invoked. Although its creaetion code is in place (loaded, installed, copied, scp/rsyn--copied, etc))
            Who is supposed to run, vs , who runs; ghcli-login.bash?

        Oh, it is used in a script, but, it cannot be used, until, another one! The latter is called , but is required by the former. "Circular dependency".
        On the other hand, we want to avoid repeat (DRY) and also for other reasons: we want to consume our own produce.

        ^ Identified a manual step, and a bigger picture about manual/interactive/cmdl/cli/shell gaps/defer s: defer to user (affordances-repertoire)

    "

    # appended fragment (keep minial)
    # the part hat is directly put into the bashrc
    HANDLER_SOURCE_SCRIPT_FRAGMENT="$(mktemp $TEMP_FOLDER/mmmXXXXXX -u)"
    cat > $HANDLER_SOURCE_SCRIPT_FRAGMENT <<EOF_STARTUP_DIRECT
        # my-tf-sctipts-are-appended  # marker to avoid appanding it again when debugging.
        cat ~/.bashrc
        echo "This is inline directly injected inside .bashrc:  \\\$\\\$=\$\$"
        pwd
        # the rest is refactored into above script, which is appended to, ... ok no: kept separte!
                source ~/scripts-sosi//scripts_to_push/dot_bashrc.bash
        #
        source ~/scripts-sosi/scripts_to_push/dot_bashrc.bash
        source ~/scripts-sosi/dynamically-generated-replaced.source.bash
        #
EOF_STARTUP_DIRECT

    # only if not already appended?
    verify_appended_remote_bashrc || {

        # A pipe, literally across the ocean!
        cat "$PIPE_TEMP_FILEN" | \
        ssh $SSH_CLI_OPTIONS  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" \
            "bash -c 'set -eux; cat > ~/scripts-sosi/dynamically-generated-replaced.source.bash'"

        # A pipe, literally across the ocean!
        cat "$HANDLER_SOURCE_SCRIPT_FRAGMENT" | \
        ssh $SSH_CLI_OPTIONS  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" \
            "bash -c 'set -eux; cat >> ~/.bashrc'"
    }

    rm "$PIPE_TEMP_FILEN"
    rm "$HANDLER_SOURCE_SCRIPT_FRAGMENT"
}


# Not used:
function local_names_shell_env {
    # for local shell (interactive)
    # todo: source this, somehow
    # history: shell_env, local_names_shell_env
    export UP="$(pwd)/common/localmachine/up-matmul-cuda-experiment.bash"
    echo '$UP bash'
    # etc
    echo "WARNING. NOT IMPLEMTNTED"
    return 1
}

function remote_names_env {
    # for local scripts
    # envs for filenames, foldr names, etc
    # history: remote_names, remote_names_env

    # inconsistency: non-crystal usage / dataflow
    SSH_CLI_OPTIONS='-i ~/.ssh/paperspace_sosi_fromlinux'
    PAPERSPACE_IP="$(cd "$TF_MAIN_TF_DIR" ; terraform output -raw public_ip_outcome)"
    PAPERSPACE_USERNAME="$(cd "$TF_MAIN_TF_DIR" ; terraform output -raw username_outcome)"
    export SSH_CLI_OPTIONS PAPERSPACE_IP PAPERSPACE_USERNAME
    echo "::: $SSH_CLI_OPTIONS $PAPERSPACE_IP $PAPERSPACE_USERNAME" > /dev/null

}

function verify_appended_remote_bashrc {

    remote_names_env
    ssh $SSH_CLI_OPTIONS  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" \
        "bash -c 'cat ~/.bashrc | grep my-tf-sctipts-are-appended'"
}


function ssh_go_into_shell {

    remote_names_env

    # implicitly, runs "bash"
    # implicitly runs .bashrc, hence, the chain of others: dot_bashrc.bash, dynamically-generated-replaced.source.bash
    ssh $SSH_CLI_OPTIONS  "$PAPERSPACE_USERNAME@$PAPERSPACE_IP"
}

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
    # echo -ne "yes\n\n\n" | \  # did not work
    ssh $SSH_CLI_OPTIONS  \
        -o StrictHostKeyChecking=no  \
        "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" "sudo touch /i-was-here"


    # From: ... (?)
    : "Note:
            This option controls the behavior of SSH when connecting to a new host for the first time or if the host key has changed.
            Disabling StrictHostKeyChecking and using /dev/null for UserKnownHostsFile means that you are not verifying the host's identity,
            (otherwise) prevents "man-in-the-middle" attacks.
            "


    echo 1>&2 'You need to do this manually in interactive (shell/bash terminal mode) on remote machine: (which in tuen may have more manual steps)
✋     bash ~/scripts-sosi/scripts_to_push/inception_script_manual.bash
    '
    # but, there is a lot run after this ^. Sine there is a whole dot_bashrc chain (with its own layers/orbits).
    # Note: I have already said this here:
    #    provisioning_scripts/terraform/common/localmachine/upload_scripts_to_there.bash
    # todo: push in something, for "affordances repertiore"!, and appear just in the end before the prompt.
    # But then, It will need a whole scripting-system for that! (modular, resuable outside this context!)
    # oh, there is this "new group" thing too! (in between).
    #   and it is sombeh-idem-potent. (single-potent! one-track-mind?)

    export SSH_CLI_OPTIONS
    # export -f scp_file
    # upload_scripts
    bash "$FROMLOCAL_SCRUPTS/upload_scripts_to_there.bash"
    # ^ also installs? (they should be seaprated. small code-block scripts?)

    append_remote_bashrc

    ssh_go_into_shell

    echo "_____________"
    echo 1>&2 "show_outputs: finished the happy path."

}

function ________subcommand___bash {

    echo "Runs an interactive subshell in the remove machine UP-ed by terraform"

    cd "$TF_MAIN_TF_DIR" && pwd
    echo -e "\n\n\n Your interactive subshell:"
    # PATH="$PATH"


    : || {
    # SKIPPING OLD version 0.0.4
    # Cancelled attempt 1
    bash -c "$(cat < < 'EOF_STARTUP'... EOF_STARTUP)"

    # Cancelled attempt 2: (SKIPPING)
    # Putting the prompt π4
    # replaced by "dynamically-generated-replaced.source.bash"
    # It is already put there in the .bashrc (may be more than once)
    cat >> ~/.bashrc <<'EOF_STARTUP'
                # appended
                PS1=...
                # ... same as dynamically-generated-replaced.source.bash
EOF_STARTUP
                # exec bash --norc # not norc
                # exec bash
                exec bash
                ssh_go_into_shell
    }

    # Since I already updated the bashrc there: already ran the "append_remote_bashrc"
    # no need for the `bash -c ".... ; exec bash"` shenanigans.
    verify_appended_remote_bashrc
    # Implicitly, also runs the chain from .bashrc:
    ssh_go_into_shell

    echo "_____________"
    echo "good."

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




function ________subcommand___rsync2 {
    echo "rsync both ways"

    remote_names_env

    #export REPO_ROOT="$HOME/gpu-experimentations"
    #export EXPERIMENT_DIR="$REPO_ROOT/experiments/11_matrix_cuda"
    #SSH_CLI_OPTIONS='-i ~/.ssh/paperspace_sosi_fromlinux'
    #PAPERSPACE_IP="$(terraform output -raw public_ip_outcome)"
    #PAPERSPACE_USERNAME="$(terraform output -raw username_outcome)"
    # REMOTE_USER="$PAPERSPACE_USERNAME"
    # REMOTE_HOST="$PAPERSPACE_IP"

    # EXPERIMENT_DIR, WDIR
    # WDIR="$(pwd)"
    _CWD="$(pwd)"
    WDIR="${_CWD/#$HOME/~}"
    # CLONEBASE, REMOTE_BASEDIR, REMOTE_REPOROOT
    LOCAL_REPO_ROOT="$REPO_ROOT"
    # REMOTE_REPOROOT="$REPO_ROOT"
    # REMOTE_REPOROOT="${REPO_ROOT/#$HOME/~}"
    # export REMOTE_REPOBASE="/home/$PAPERSPACE_USERNAME/workspace"
    export REMOTE_HOME="/home/$PAPERSPACE_USERNAME"
    export REMOTE_REPOBASE="$REMOTE_HOME/workspace"  # delay/defer/lazy evaluation
    echo "$REMOTE_REPOBASE"
    export REMOTE_REPOROOT="$REMOTE_REPOBASE/$(basename $LOCAL_REPO_ROOT)"
    echo "assert $WDIR $LOCAL_REPO_ROOT $REMOTE_REPOROOT $REMOTE_REPOBASE"  > /dev/null

    : || {
    # set -xue
    set +x
    echo

    scrupt="set -eux; echo 'hi' ; sudo mkdir -p \"$REMOTE_REPOROOT\" && cd \"$REMOTE_REPOROOT\" && git clone git@github.com:sohale/gpu-experimentations.git \"$REMOTE_REPOROOT/..\" && mkdir -p \"$WDIR\"; cd \"$WDIR\"; pwd; ls -alth "
    echo $scrupt
    exit 1

    ssh $SSH_CLI_OPTIONS \
        "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" \
        'echo $HOME'
    exit 22
    }

    # Analysis report: In below, use which:
    #    *    source dot_bashrc.bash
    #    *    source ~/.bashrc
    # anser: source dot_bashrc.bash


    export GITCLONE_SSHURL="git@github.com:sohale/gpu-experimentations.git"

    ssh $SSH_CLI_OPTIONS \
        "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" \
        bash -c ":&&\
            set -x &&\
            source $REMOTE_HOME/scripts-sosi/scripts_to_push/dot_bashrc.bash  &&\
            set -eux &&\
            echo 'hi' &&\
            sudo mkdir -p \"$REMOTE_REPOBASE\" &&\
            sudo chown -R \"\$USER:\$USER\" \"$REMOTE_REPOBASE\" &&\
            cd \"$REMOTE_REPOBASE\" &&\
            { : || DONT || mv \"$REMOTE_REPOROOT/\" \"$REMOTE_REPOBASE/hij\" || : ; } &&\
            { ls \"$REMOTE_REPOROOT/\" || git clone $GITCLONE_SSHURL \"$REMOTE_REPOROOT\"; } &&\
            mkdir -p \"$WDIR\" &&\
            cd \"$WDIR\" &&\
            pwd &&\
            ls -alth &&\
            :
        "

    LOCAL_INDICATOR="🅛Local"
    #  🅛 🅡  Ⓛ Ⓡ
    echo "$LOCAL_INDICATOR: Now ready for rsync from Ⓛ to Ⓡ"

    # Ensure the timezone is the same on both systems
    echo "Checking timezones..."
    LOCAL_TZ=$(timedatectl show --property=Timezone --value)
    REMOTE_TZ=$(ssh $SSH_CLI_OPTIONS "$PAPERSPACE_USERNAME@$PAPERSPACE_IP" "timedatectl show --property=Timezone --value")

    echo " Ⓛ $LOCAL_TZ  Ⓡ $REMOTE_TZ"

    if [[ "$LOCAL_TZ" != "$REMOTE_TZ" && "$LOCAL_TZ" != "Etc/$REMOTE_TZ" ]]; then
        echo "WARNING: Timezones are different! Local: $LOCAL_TZ, Remote: $REMOTE_TZ"
        echo "Syncing files might cause unexpected timestamp issues."
        exit 1
    fi

    # The --delete flag removes files from the destination if they no longer exist in the source.

    echo "1: Syncing from Local → Remote... (remove from remote if necessary)"
    rsync \
        -e "ssh $SSH_CLI_OPTIONS" \
        --delete \
        -avz --progress --times --perms --owner --group \
        --exclude=".git" --exclude="*.swp" --exclude="*.bak" \
        "$LOCAL_REPO_ROOT/" "$PAPERSPACE_USERNAME@$PAPERSPACE_IP:$REMOTE_REPOROOT/"
    echo "Sync (part 1) complete!"
    exit 1

    echo "2: Syncing from Remote → Local... (don't remove (local?) if something in remote is removed)"
    SUBFOLDER="experiments/11_matrix_cuda"
    rsync \
        -e "ssh $SSH_CLI_OPTIONS" \
        -avz --progress --times --perms --owner --group \
        --exclude=".git" --exclude="*.swp" --exclude="*.bak" \
        "$PAPERSPACE_USERNAME@$PAPERSPACE_IP:$REMOTE_REPOROOT/$SUBFOLDER/" "$LOCAL_REPO_ROOT/$SUBFOLDER/"
    echo "Sync complete!"

    # todo: now, you can remote-bash nvcc ... too (NVidia compilation script, etc)
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
