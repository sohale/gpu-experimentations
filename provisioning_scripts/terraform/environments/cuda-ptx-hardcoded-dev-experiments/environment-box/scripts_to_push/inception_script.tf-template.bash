#!/bin/bash

# this is a terraform template, i.e. the parts  `\$\{var. ... }` in this file (I cannot write them without escaping, or maybe ${input_public_ip}, even in the comments) will be replaced by terraform

# This is executed in rioot user: by tf provisioning.
# Avoid `$ {}` syntax of bash, since it is template-instantiated by terraform.
echo -e "\n [$0] \n$(date)" >> ~/.sosi-footprints.log
# ~/.sosi-footprints

# Not the startup.
# Provisioning. Creation, Genesis, Inception
# By terraform:
# resource "paperspace_script" "my-startup-script-1" {

# Different to  local_manual__setup_at_creation.bash . That is from direct scripts

# environment_boxes/neurotalk/scripts_to_push/inception_script_tf.bash
# environment_boxes/neurotalk/scripts_to_push/docker_user_fix.bash
# environment_boxes/neurotalk/local_manual__setup_at_creation.bash


      # We are sudo, I suppose.

      # affordance-emiting part: "my-affordance-script"
      # the static part. The `run_once` is better be "true", but happns to work also when  `run_once=false`

      ADIR="/sosi/instance-info"
      mkdir -p "$ADIR/seedinput/"
      mkdir -p "$ADIR/derived/"
      mkdir -p "$ADIR/debug/"

      echo "${input_public_ip}" >> "$ADIR/seedinput/public_ip"

      # Causes a Cycle error:
      # echo "$ {paperspace_machine.my-gpu-machine-1.public_ip_address}" >> "$ADIR/derived/public_ip_address"
      # echo "$ {paperspace_machine.my-gpu-machine-1.orivate_ip_address}" >> "$ADIR/derived/private_ip_address"

      PING_FILE="$ADIR/debug/$(date +%Y-%m--%d-%H-%M-%S)--$RANDOM.ping.txt"
      touch "$PING_FILE"

      # Redirect all output and errors to the log file
      LOG_FILE="$PING_FILE"  # ../logfile.log"
      exec > >(tee -a "$LOG_FILE") 2>&1

date >> ~/.sosi-footprints.log

WEBSERVER_DIR="/var/www/html"
sudo mkdir -p "$WEBSERVER_DIR/"
sudo chown -R $(whoami):$(whoami) /var/www/html

INDEX_HTML="$WEBSERVER_DIR/index.html"
echo "Hello, World," > $INDEX_HTML
# Note: input_public_ip is baked into it by `templatefile()`, not by bash:
echo " from public-ip: ${input_public_ip}" >> $INDEX_HTML
echo " from whoami: $(whoami)" >> $INDEX_HTML

date >> ~/.sosi-footprints.log

ufw allow 8080
sudo ufw allow 22  # keep allowing ssh too
echo "y" | \
sudo ufw enable  # needed too. But need to whitelist ssh
sudo ufw status
nohup busybox \
    httpd \
      -f -p 8080 \
      -h "$WEBSERVER_DIR" \
      &

date >> ~/.sosi-footprints.log

echo "cat $ADIR"/'**/*'
echo cat $INDEX_HTML
echo "http://${input_public_ip}:8080/"

# ^ only works correctly if `run_once=false` (e.g. runs each time the machine is turned)
# especially for "httpd"

date >> ~/.sosi-footprints.log
echo >> ~/.sosi-footprints.log
