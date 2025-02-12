output "api_key" {
  value     = var.api_key
  sensitive = true
}


# outcome, manual, dynamic, static, etc, whatever
# "output"s = "outcome"s ?
# instance_ip
output "public_ip_outcome" {
  description = "The manually assigned public IP address, or, The public IP address of the created GPU instance"
  # value       = var.input_public_ip
  # value = paperspace_machine.my-machine.public_ip
  value = paperspace_machine.my-gpu-machine-1.public_ip_address
  # Other fields:
  #   = paperspace_machine.my-gpu-machine-1.  private_ip_address        === (known after apply)
  #   = paperspace_machine.my-gpu-machine-1.  public_ip_address         === (known after apply)

}

output "username_outcome" {
  # The Linux username
  description = "The linux username which you can use to `ssh` into the machine as `@` in your ssh command"
  # This seem shard-coded, but this allows to go beyond providers.
  #   for: ssh -v -i ~/.ssh/paperspace_sosi_fromlinux paperspace@74.82.28.237
  # hard-coded for paperspace provider, only.
  value = "paperspace"
}

# rename: keypairname_outcome
output "keypair_outcome" {
  description = "the keypair used"
  # an output (_outcome) version is needed for:
  #    generaing part of the arg of the ssh command: ssh  -v -i ~/.ssh/paperspace_sosi_fromlinux paperspace@74.82.28.237
  # regardless of the method, keep a safe output here

  # hard-coded, but will be refactored. (the 'username_outcome' will remain hard coded, unlike this)

  # hardwire it (not hard-code)
  value = var.ssh_keypair_name_inputparam
}


# todo: rename: machine? instance? real_machine? v_machine? machine_?
# Either:
# * instance_id
# * applied_info.gpu_machine_id
output "instance_id" {
  description = "The ID of the created GPU instance"
  value       = paperspace_machine.my-gpu-machine-1.id
}

output "machine_tuple_used" {
  # machine_tuple machinetype_used
  # *_used is not _outcome: = vars vs. outputs, respectively.
  description = "The machine type & OS image template used"
  # value       = (vars.machine_type, vars.machine_image_template)
  value = [var.machine_type, var.machine_image_template]
}

output "applied_info" {
  # a mixture of _used & _outcome
  # applied_debug_info, applied_debug_outcome, applied_info

  # group these into a named tuple
  value = {
    machine_tuple    = [var.machine_image_template, var.machine_type]

    # For use as filename. e.g. "C1-t04azgph"
    machine_tuple_uniquename    = "${var.machine_type}-${var.machine_image_template}"

    # public_ip_outcome, username_outcome (a "linux username"), (keypair_outcome)
    access_tuple    = [paperspace_machine.my-gpu-machine-1.public_ip_address, "paperspace"]

    # unique, good for using as part of filenames that save into / env
    # rename: gpu_machine_id
    instance_id      = paperspace_machine.my-gpu-machine-1.id

    # user_debug, not "linux username"
    tfuser            = data.paperspace_user.lead-engineer-user
  }
}

# debugging and tracing
output "🐞 user_data" {
  value = data.paperspace_user.lead-engineer-user
}