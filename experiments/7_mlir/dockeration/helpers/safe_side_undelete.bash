
# mount
lsblk -f
df -T


sudo apt-get install \
 extundelete \
 testdisk \
 photorec \
 e2fsprogs



# usage

#sudo extundelete /dev/sdX --restore-file path/to/deleted/file
#sudo extundelete /dev/sdX --restore-directory path/to/deleted/directory
#sudo extundelete /dev/sdX --restore-all

# sudo testdisk   # Choose the partition table type (usually Intel for PCs).

# sudo debugfs -w /dev/sdX


# unmounts
# sudo umount /dev/vda1
# sudo extundelete /dev/vda1 --restore-all
# # save in: ./RECOVERED_FILES
