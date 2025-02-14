echo "ghcli-login.bash"
set -eu
set -x

# workaround
cp ~/secrets/ghcli-token-1.txt ~/secrets/ghcli-token.txt

function create_a_whole_new_ssh {
  # not tested

    local KEYPAIR_NAME="${1}"
    local MY_USER_EMAIL="sohale@gmail.com"

    # local SSH_KEY_PATH="~/.ssh/id_ed25519"
    local SSH_KEY_PATH="$HOME/.ssh/$KEYPAIR_NAME"

    mkdir -p ~/.ssh

    # local COMMENT="comment: $MY_USER_EMAIL"
    local COMMENT="$MY_USER_EMAIL"

    ls -alth $SSH_KEY_PATH || \
    ssh-keygen -t ed25519 -C "$COMMENT" \
        -f "$SSH_KEY_PATH" \
         -P ""

    test -f $SSH_KEY_PATH  || exit 1  # without ""
    test -f $SSH_KEY_PATH.pub || exit 1 # without ""

    # on remote machine, but, at creation time. Hence, change the name of the script.
    # needs source ing?
    eval "$(ssh-agent -s)"
    ssh-add "$SSH_KEY_PATH"

    local ENV_FILE="/home/paperspace/scripts-sosi/refresh_ssh_agent.env"
        # redo_env.bash.source.env
        # This script also can run each time when we log in (as ~/.bashrc ?).
        # not yet. But can be a thought. I will think about it.
        # A minumal execuison context, to export this funciton/ subshell script
        # Dont use it for wider scope such as a dot-bashrc
        # You can have nother one alongside ot for otehr things. So, there will be three:
        #/home/paperspace/scripts-sosi/
        #      refresh_ssh_agent.env    # super minimal
        #      env_context.bash         # semi-monimal
        #      scripts_to_push/dot_bashrc.bash   # ok, but dont make it complicated. NOTE: I think this is no longer used.

    echo "
        eval \"\$(ssh-agent -s)\"
        ssh-add $SSH_KEY_PATH
    " \
        > $ENV_FILE

    echo "
    Please 🫱 manually run:

    $(cat $ENV_FILE)

    Or simply

    source $ENV_FILE

    "

    # not yet!
    # ssh -T git@github.com  || [ $? -eq 1 ]
}

function solution1 {
# use (or create) a key pair
# and login tgoo
# and authebtitace (auth login) as well
# uploads to your (personal) profile on github

local GITHUB_PAT_TOKENFILE=~/secrets/ghcli-token.txt
local KEYPAIR_NAME="github_sosi_from_paperspace"


local SSH_KEY_PATH="$HOME/.ssh/$KEYPAIR_NAME"
# uploads to your (personal) profile on github
gh ssh-key add "$SSH_KEY_PATH.pub" \
    --title "GitHub-CLI-yet-another-retitle"

export GH_TOKEN=$(cat "$GITHUB_PAT_TOKENFILE")

gh auth login --with-token

: || \
gh auth login  --with-token \
    --hostname github.com \
    --git-protocol ssh \
    < "$GITHUB_PAT_TOKENFILE"


ssh -T git@github.com  || [ $? -eq 1 ]

# now you are able to git clone
}


# (originally) intended solution, but has not worked yet
function solution2 {
    # I want to avoid  creating or using a key pair
    # my preferred solution

    # not tested

    local GITHUB_PAT_TOKENFILE=~/secrets/ghcli-token.txt

    # gh auth login --with-token


    # --skip-ssh-key --> Skip generate/upload SSH key "prompt"
    # of course we dont want a "prompt" here, regardless of the

    gh auth login  --with-token \
        --hostname github.com \
        --git-protocol ssh \
        --skip-ssh-key \
        < $GITHUB_PAT_TOKENFILE


    ssh -T git@github.com  || [ $? -eq 1 ]

    # now you are able to git clone
}

function solution3 {
    # needs to be source d (NO!)
    # hence, not good
    export GH_TOKEN=$(cat ~/secrets/ghcli-token.txt)
    # also we can use a baked&laced env file
    # gh auth login --with-token

    gh auth login --with-token


    ssh -T git@github.com  || [ $? -eq 1 ]

    # now you are able to git clone
}


# OK Now I know I was dealing with a paradox!
# I wanted to pull but using PAT, amd not ssh
# I need tp read about PAT & also "secret store s
function solution4 {
    # why was not wrkig before
    #export GH_TOKEN=$(cat ~/secrets/ghcli-token.txt)
    #gh auth login --with-token <<< "$GH_TOKEN"
    gh auth login --with-token <<< "$(cat ~/secrets/ghcli-token.txt)"
    ssh -T git@github.com  || [ $? -eq 1 ]

    # now you are able to git clone
}

# note: has inforsource (antisink/unsink)
function solution5_ssh {
    # back to senses
    # need ssh becaise I awnt to git clone using ssh
    # First uses PAT to upload the ssh key, then uses that ssh key.
    # uploads to your (personal) profile on github


    local KEYPAIR_NAME="github_sosi_from_paperspace"
    local GITHUB_PAT_TOKENFILE=~/secrets/ghcli-token.txt

    create_a_whole_new_ssh  $KEYPAIR_NAME

    gh auth login --with-token <<< "$(cat "$GITHUB_PAT_TOKENFILE")"

    local SSH_KEY_PATH="$HOME/.ssh/$KEYPAIR_NAME"

    # add to up
    # but which key?! "current gh" ssh keypiar-name?
    gh ssh-key add $SSH_KEY_PATH.pub \
        --title "provisioned-paperspace-gpu-1" \
        --type authentication

    gh auth status

    # Now that we set up the keypair, let's use it (instead of that PAT)
    gh auth login  --with-token \
        --hostname github.com \
        --git-protocol ssh \
        < $GITHUB_PAT_TOKENFILE

    gh auth status


    # Displays the status
    # Here? (git global) or here (gh state)  or ssh (default ssh) or there (?), or repo (git local) or repo (gh local)
    gh auth switch

    gh auth status


    # returns the status of the remote command, otherwise, 255.
    # So, as long as the exit code is not 255, it is OK (accept "1" only)
    ssh -T git@github.com  || [ $? -eq 1 ]


    # now you are able to git clone
}

# solution1
# solution2
# solution3  # needs to be source d (NO, it doesn't)
# solution3
# solution4
solution5_ssh  # has inforsource (antisink/unsink)

ssh -T git@github.com  || [ $? -eq 1 ]
# now you are able to git clone
# that is the (single) goal of this script


# "Configure git to use GitHub CLI as the credential helper for all authenticated hosts"
# see https://cli.github.com/manual/gh_auth_setup-git
gh auth setup-git

# useful other affordances:
#      gh auth switch

gh auth status

echo "exports: source /home/paperspace/scripts-sosi/refresh_ssh_agent.env"
echo "Happy ending for: ghcli-login.bash"
