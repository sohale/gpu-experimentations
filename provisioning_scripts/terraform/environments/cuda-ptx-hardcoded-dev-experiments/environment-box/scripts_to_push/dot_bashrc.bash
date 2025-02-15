# I think this is no longer used

# placeholder for dot bashrc
# "The dot_bashrc.bash, an extracted, controlled, owned, part of .bashrc (a branch thread leads here)"
echo "The dot_bashrc.bash, an extracted, controlled, owned, part of .bashrc (a branch thread leads here)"

# static: git-time static (git-level static: it is pushed)
echo "This is the static script called by .bashrc:  \$\$=$$"

sudo timedatectl set-timezone UTC

# this part content is dynamic (deploy-time dynamic)
source /home/paperspace/scripts-sosi/refresh_ssh_agent.env

# no:
# things wil be appended here!
# including the above item?
# so, this fil, is actually a template
#
# instead: this is the static part (don't use multiple solutions!)
# not a good practice to change a version-controlled file
