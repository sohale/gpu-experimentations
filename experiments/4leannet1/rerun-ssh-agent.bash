
set -e
# If no "ssh-agent", launch one:
ps aux | grep -v grep | grep ssh-ag || \
eval "$(ssh-agent -s)"

ssh-add ~/.ssh/github_rsa

echo "Checking existing ssh-agents:"
ps aux | grep -v grep | grep ssh-ag || :

echo "."
