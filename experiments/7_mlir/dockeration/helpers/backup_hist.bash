set -eux

mkdir -p ~/history/backups.bash_history
sudo chmod +w ~/history
sudo chmod +w ~/history/backups.bash_history/

export suffix="a"


# history > ~/history/hist_$(date +%F)-${suffix}.history
while true; do

   hbfn=~/history/hist_$(date +%F)-${suffix}.history

   history  > $hbfn \
      &&  break  # Success

   set -eux
   # Double check if it is failing for existing
   ls -1 $hbfn  # if not exists, something else, hence, quit
   suffix=${suffix}"a"
done



# backup the .basj_history
cp ~/.bash_history  ~/history/backups.bash_history/bash_history_backup_$(date +%F_%H-%M-%S)

# Seal of protection
chmod  -R -w ~/history/*

# todo: run this in a cron
# crontab -e
# 0 0 * * * ...../backup_hist.bash

# Also make sure we have this in ~/.bashrc (which we do)
# shopt -s histappend
# PROMPT_COMMAND="history -a; history -c; history -r; $PROMPT_COMMAND"

# If in another session: history -w ~/.bash_history_recovered


# File Recovery Tools:

# like extundelete (for ext3/ext4 filesystems) or
# photorec to attempt recovery of deleted files.
# Be cautious, as continuing to use the system can overwrite the deleted file.
