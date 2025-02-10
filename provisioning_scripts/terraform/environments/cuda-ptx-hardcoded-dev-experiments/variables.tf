# Not the values, but the space holding (or not holding) these values
# inputs

variable "team_id" {
  description = "The ID of my Paperspaces team"
  type        = string
}

variable "project_id" {
  description = "The ID of my project"
  type        = string
}

variable "project_name" {
  description = "Dummy: The name of my project"
  type        = string
}

variable "api_key" {
  description = "My Paperspace API key to connect to provider"
  type        = string
}

# todo: make use of such "secret name" in the terraform code
# https://console.paperspace.com/t10oyrfnaf/projects/pudltta9b1i/settings
# is a "shared secret" (?)

variable "apikey_secret_name" {
  description = "The name (key) of the secret that has the API Key. Currently not used."
  type        = string
}

variable "github-cli-pat" {
  description = "my Github PAT to connect to my github account and repos from that remove machine. Thes string is one of the secrets, is on t local side, but then will be saved on the remote-machine's side."
  type        = string
}

variable "remote_linux_username" {
  description = "The username to connect to the remote machine"
                # So far, this is used only for copying secrets such as github-cli token
                # (using the remote-exec "provisioner")
                # during provsionsin-time (terraform-apply time!)
                # to be picked up only aftter provisionsing
  type        = string
  default     = "paperspace"  # Default user for Paperspace machines
}

variable "ssh_keypair_name_inputparam" {
  description = "the keypair used. Creared in local"
  # for: generaing part of the arg of the ssh command: e.g.
  #         ssh  -v -i ~/.ssh/paperspace_sosi_fromlinux paperspace@74.82.28.237

  type        = string

  # (not hard-coded anymore. The "default"ness sense and implication)
  # hard-coded, but will be refactored. (the 'username_outcome' will remain hard coded, unlike this)
  # default = "paperspace_sosi_fromlinux"
  # "no default". "No fucking default."
}
/*
History & Design:

Renameing history:
    ssh_keypair_param -> ssh_keypair_name_param -> ssh_keypair_name_inputparam
  ( key_pair_name )

It was previously an output (why?!) -- emphasising the "hard-codedness of it,
since it was in the code?" No, it was in the bash scripts. I see.

      output "keypair_outcome" {
        description = "the keypair used"
        # for: generaing part of the arg of the ssh command: ssh  -v -i ~/.ssh/paperspace_sosi_fromlinux paperspace@74.82.28.237

        # hard-coded, but will be refactored. (the 'username_outcome' will remain hard coded, unlike this)
        value = "paperspace_sosi_fromlinux"
      }


Then became an parameter ( ssh_keypair_param ---> ssh_keypair_name_inputparam )
I previously had written: "will be refactored". It's time now.

But we need the output, so I kept the "output" version.

PS. older comments in the .tfvars file:
        # THE FROM MACHINE

        #  # as determined in terraform/common/per-provider/paperspace/setup_anew_ssh.bash
        #  key_pair_name = "paperspace_sosi_fromlinux"
        # Use your custom SSH key pair name
#
*/

/*
# ip is an output ! not input!
# we can have onw as input to, but the outcome (being dynamic, static, none, etc) is an output
# change this one according totout input it is
# omapre with : public_ip_outcome
variable "public_ip" {
  description = "The manually assigned public IP address"
  type        = string
  default     = "184.105.217.24"
}
*/
/*
variable "public_ip" {
  description = "The public IP address"
  type        = string
  default     = ""
}
*/

variable "input_public_ip" {
  # Anything explicityl specified, input parameter.
  description = "The manually assigned public IP address"
  type        = string
}



variable "region_pdefault" {
  description = "Region to deploy the instanc?? (when region_parameter is not speciied)" # in the prociver block
  type        = string
  #default     = "AMS1" ?"eu-west-1"  # Europe region
  #"East Coast (NY2)"
}
variable "region_parameter" {
  description = "Region to deploy the instance??. Is optional (in papersapce), defaults to provider region if not specified."
  type        = string
  #default     = "AMS1" ?"eu-west-1"  # Europe region
  # "East Coast (NY2)" // optional, defaults to provider region if not specified
}

// "instance_type" = machine_type
variable "machine_type" {
  description = "Type of GPU instance to deploy"
  type        = string
  # default     = "A5000"
}
# rename to : machine_name
variable "instance_name" {
  description = "Name of the GPU instance"
  type        = string
  # default     = "A5000-06152024"
}
/*
variable "instance_image" {
  description = "Image ID for Ubuntu 22.04"
  type        = string
  default     = "ubuntu-22.04"  # You may need to find the exact ID for Ubuntu 22.04
}
*/
variable "instance_disk_size_gb" {
  description = "Disk size for the instance in GB"
  type        = number
  default     = 50
}


variable "machine_image_template" {
  description = "Machine's OS / image Templates"
  type        = string
  # default     = "t04azgph"
}


# ########### Image / OS ###########
# ways to name it:
#  machine_image_template
# pspace os-template
# nvidia-cuda-vm-machine-species-template
# A psecifi species of machine !
# Machine Templates ?
# template: https://registry.terraform.io/providers/paperspace/paperspace/latest/docs/resources/machine
#
# machine_image_template =
#  = "t04azgph" // this is one of the Ubuntu Server 18.04 templates

/*
For choices: pspace os-template list
  tz0ireoj  prod-u20-gpu-240606       Ubuntu 20.04 Desktop                 null

  tilqt47t  mliab-u22-240117-1x-h100  Ubuntu 22.04                         100
  tqqsxr6b  mliab-u22-240117-1x-a100  Ubuntu 22.04                         100
  t7vp562h  mliab-u22-240116-h100     Ubuntu 22.04                         100
  tvimtol9  mliab-u22-240116          Ubuntu 22.04                         100

  mliab=? MLiaB  ml_in_a_box_22_04.sh
*/

/*
tz0ireoj  prod-u20-gpu-240606       Ubuntu 20.04 Desktop                 null

tilqt47t  mliab-u22-240117-1x-h100  Ubuntu 22.04                         100
tqqsxr6b  mliab-u22-240117-1x-a100  Ubuntu 22.04                         100
t7vp562h  mliab-u22-240116-h100     Ubuntu 22.04                         100
tvimtol9  mliab-u22-240116          Ubuntu 22.04                         100
twnlo3zj  mliab-u20-240606          Ubuntu 20.04                         null
t9taj00e  centos-220817             CentOS 7 Server                      null
*/


# ############# Machine type ################

# https://docs.digitalocean.com/products/paperspace/machines/details/features/
# https://docs.digitalocean.com/products/paperspace/machines/details/pricing/

/*
GPU+ (M4000)
P4000
P5000
P6000
RTX4000
RTX5000
A4000
A5000
A6000
V100
V100-32G
A100
A100-80G
H100
*/

# "Invalid machine type: tvimtol9"
