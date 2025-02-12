# Is this used? yes, but not in the scirpts.
# Is this deprecated? No. But it is not essential.
# Manual utility for generating a new ssh key pair for: `ssh` and github access. Example: `paperspace_sosi_fromlinux`
# This automates creating of key-pairs.
# Forked from: /myvol/pocs_for_nikolai/terraform/common/per-provider/paperspace/setup_anew_ssh.bash
# But the script itself is safe to run since everyting is a dry-run, via '_dr'
# (dry-run mode)

echo "Only run manually.
Comment, Change SSHKEYPAIR_NAME temporarily, change (dry-run mode).
run this script,
and revert & un-comment again the `exit` command.
"
exit 1  # uncomment and comment back



luxargs luxargs luxargs luxargs echo
: || ```txt
  luxargs
     luxargs
       luxargs
         echo   ↲
  luxargs
     luxargs
       echo   ↲
  luxargs
     echo   ↲
  echo   ↲
```


dryrun='luxargs echo '
_dr="${dryrun}"


$_dr \
MY_EMAIL="your_email@example.com"

# Name is deliberately changed to a dummy name to avoid overwriting pre-existing keys.
# SSHKEYPAIR_NAME="paperspace_sosi"
# SSHKEYPAIR_NAME="paperspace_sosi_fromlinux"
SSHKEYPAIR_NAME="dummyxyzxyz"

# Slightly safer, to protect if such key-pair already exists
test ! -f ~/.ssh/${SSHKEYPAIR_NAME}
test ! -f ~/.ssh/${SSHKEYPAIR_NAME}.pub
$_dr \
ssh-keygen -t rsa -b 4096 \
  -C "${MY_EMAIL}" \
  -f ~/.ssh/${SSHKEYPAIR_NAME}

luxargs $_dr \
echo '
eval "$(ssh-agent -s)"
'

$_dr \
ssh-add ~/.ssh/${SSHKEYPAIR_NAME}

echo
$_dr \
cat ~/.ssh/${SSHKEYPAIR_NAME}.pub

# To print on screen, and also to copy to clipboard
echo
echo "This is your public key. Copy this to clipboard, and paste it to github."
echo "       Original location: ~/.ssh/${SSHKEYPAIR_NAME}.pub"
$_dr \
echo "\"$(cat ~/.ssh/${SSHKEYPAIR_NAME}.pub)\""


echo "Update the up.bash script with the name. (Also a deprecated tfvar)
key_pair_name = \"${SSHKEYPAIR_NAME}\"
"
