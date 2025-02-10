


data "paperspace_template" "nvidia-cuda-vm-machine-species-template" {
  // A psecifi species of machine !
  id = var.machine_image_template
}

# Thos creates a user in the team (?)
data "paperspace_user" "lead-engineer-user" {
  email   = "sohale@gmail.com"
  # This is a personal user, not a team user.
  # Personal ones should not define "team_id"
  # team_id = var.team_id
}

# see above (paperspace_user), just for debugging. Keep this here.
# output "user_debug" {
#   value = data.paperspace_user.lead-engineer-user
# }


resource "paperspace_script" "my-startup-script-1" {
  # resource "resource type" "is of the resource instance" { ... }

  name        = "Start-up script on Ubuntu"
  description = "Start-up boot-time script, on Ubuntu Machine to enable access from: httdp, ssh. And run: httpd (Apache http server). Confgure: A simple page. Impleemntaiton details: allow 8080 in `ufw` firewwall. This script is run once (?) after creating a server machine (not a login shell startup, and not per-power-on startup. But per-deploy-create-vm-machine. )."


  script_text = templatefile(
    // "${path.module}/inception_script.tf-template.bash"
    "${path.module}/environment-box/scripts_to_push/inception_script.tf-template.bash"
    // should I change ${path.module}? Previously, it was a symbolic link.
    , {
      input_public_ip = var.input_public_ip
    }
  )
  # "${path.module}/inception_script.tf-template.bash"
  # "/mnt/volume_lon1_01/pocs_for_nikolai/environment_boxes/neurotalk/scripts_to_push/inception_script.tf-template.bash"
  #  "${path.module}/inception_script_tftemplate.bash", {...
  #  "${path.module}/inception_script_tftemplate.bash", {
  # script_text = file("${path.module}/inception_script.tf-template.bash")
  # script_text = file("${path.module}/inception_script_tf.bash")
  # ^ This is a symblic link, refering tio the real file, in the folder in which it is version-vontroeled, near other scripts related to this prvisioning in one place.
  # path.module is "."
  # Moved to:
  #script_text = file("${path.module}/../../environment_boxes/neurotalk/scripts_to_push/inception_script_tf.bash")
  # todo: which:
  # "${path.module}/scripts_to_push/inception_script_tf.bash"
  # "${path.module}/environment_boxes/neurotalk/scripts_to_push/inception_script_tf.bash"
  #
  # : <<COMMENT
  # #  script_text = <<EOF
  # #!/bin/bash
  # COMMENT
  # : <<COMMENT
  # EOF
  # COMMENT

  # It could be disabled:
  is_enabled  = true

  #   run_once = true: The script will run only once after the machine is initially created.
  #   run_once = false: The script will run every time the machine is started.
  run_once    = false
}
# for ssh:
# https://console.paperspace.com/account/settings/ssh-keys
# https://docs.digitalocean.com/products/paperspace/machines/how-to/connect-using-ssh/
# "paperspace_sosi_fromlinux_pub"

resource "paperspace_machine" "my-gpu-machine-1" {
  region           = var.region_parameter
  name             = var.instance_name
  machine_type     = var.machine_type
  size             = var.instance_disk_size_gb
  billing_type     = "hourly"

  assign_public_ip = true
  # optional, remove if you don't want a public ip assigned
  # false: ??
  # paramertise?

  template_id               = data.paperspace_template.nvidia-cuda-vm-machine-species-template.id
  user_id                   = data.paperspace_user.lead-engineer-user.id // optional, remove to default
  team_id                   = data.paperspace_user.lead-engineer-user.team_id
  script_id                 = paperspace_script.my-startup-script-1.id
  shutdown_timeout_in_hours = 1
  # live_forever = true # enable this to make the machine have no shutdown timeout

  /*
  # as determined in terraform/common/per-provider/paperspace/setup_anew_ssh.bash
  key_pair_name = "paperspace_sosi_fromlinux"
  */

  /*
   # Add tags for better resource management
  tags = {
    # How can this conttribute?
    "Environment" = "Development"
    "Project"     = "MyProject"
    "Owner"       = "YourName"
  }
  */

  # `provisioner "local-exec"`
  #     How may times a  {will it be executd?
  # also : `local`

  # This is how we take into account the "public_ip" (input_public_ip) variable
  # todo: Check if the paperspace_machine resource in the Terraform provider for Paperspace supports a way to assign an existing public IP directly, you should use that feature
  provisioner "local-exec" {
    # Error running command '...' Permission denied
    command = "echo ${var.input_public_ip} | sudo tee /etc/public_ip"
  }

}

# This approach was not used (to copy a secret to a remote machine).
# Instead, the secret was passed as a variable to the script that was run on the remote machine.

# todo: "secrets/github_pat.txt" --> "secret_laced_envfiles" -> "secrets/secretes.env"
# one folder for all, even one file for all, maybe

resource "null_resource" "copy_github_cli_pat" {
  depends_on = [paperspace_machine.my-gpu-machine-1]

  provisioner "remote-exec" {
    connection {
      type     = "ssh"
      user     = var.remote_linux_username
      # not input_public_ip, nad not the output (despite )
      #host     = paperspace_machine.my-gpu-machine-1.public_ip
      #host     = outputs.public_ip_outcome
      #host     = var.public_ip_outcome

      # Neither variable (input), nor "output" (output), but the "resource field"s in the middble
      host      = paperspace_machine.my-gpu-machine-1.public_ip_address

      private_key = file("~/.ssh/${var.ssh_keypair_name_inputparam}")
    }

    # Bakes a script "laced" with the secret "PAT" and send it away. Is it safe?
    inline = [
      "mkdir -p /home/${var.remote_linux_username}/secrets/",
      "echo '${var.github-cli-pat}' > /home/${var.remote_linux_username}/secrets/github_pat.txt",
      "chmod 600 /home/${var.remote_linux_username}/secrets/github_pat.txt"
    ]
    # Notes on 600:
    #     600 =  (owner:,group:,others:) = ("rw") ("") ("")
    #     "rw-" = 4 (read r) + 2 (write w)
    #     "---" = (no permissions)
    #     600 = "u=rw,go="
    #     (u:,g:,o:) === (owner:,group:,others:)
    #     600 =  (u:,g:,o:) = (owner:,group:,others:) = ("rw-") ("---") ("---")
    #     600 = "Only the owner can see it, ie(keep it private)".

   # todo: send an "output" about hte list of files put there ...
  }
}
