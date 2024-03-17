
set -e

# Kill "ssh-agent"s if any:
ps aux | grep -v grep | grep ssh-ag &&
ps aux | grep -v grep | grep ssh-ag \
| awk '{print $2}' | xargs kill

#eval "$(ssh-agent -s)"

#echo "slleep 1"
#sleep 1
# Why do I need to run this manually?
#ssh-add ~/.ssh/github_rsa

echo "Checking existing ssh-agents:"
ps aux | grep -v grep | grep ssh-ag || :

echo "Manually run the folllwing:"

echo 'eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_rsa'

echo
echo "."
