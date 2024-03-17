
set -ex
ps aux|grep ssh-ag
ssh-add ~/.ssh/github_rsa
eval "$(ssh-agent -s)"
