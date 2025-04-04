# Team:
# > Paperspace
# https://console.paperspace.com/entobrain/settings/profile
# ( CORE? , Gradiaent, Paperspace)
# name=entobrain
# U=entobrain
# ID=tqpio01i96

# team_u = "entobrain"


/* No team_id
# It can be instead, dynamically fetched from user: data.paperspace_user.lead-engineer-user.team_id
#    i.e. data.paperspace_user.lead-engineer-user.team_id is "tqpio01i96"
team_id="tqpio01i96"
# Can it use programmatically here? No: Variables not allowed.
# team_id=data.paperspace_user.lead-engineer-user.team_id
*/

/*
project_id="p0k609uoxdl"
project_name = "entobrain-dl1"
project_apikey_secretname="entobrain_apikey"
*/
/*
 It is meaningless to parametrise this:
region_pdefault="Europe (AMS1)"
*/

input_public_ip="9.9.9.9"

instance_name = "Physical A5000 (by Terraform)"


# machine_image_template = "twnlo3zj"  # (A5000?)
# machine_image_template = "t04azgph"  # Ubuntu Server 18.04
# machine_type="A5000"
# machine_type= "C1"

# pair: ("VM type", "template")
machine_image_template = "t17d1a6i"
machine_type="A5000"

# Use "t17d1a6i". Avoid "twnlo3zj" since it is ubuntu 20.04:
# tvimtol9  mliab-u22-241101          Ubuntu 22.04                         100
# twnlo3zj  mliab-u20-241101          Ubuntu 20.04                         null
# t17d1a6i  prod-u22-gpu-241021       Ubuntu 22.04 Desktop                 null
# t0nspur5  prod-u22-cpu-241018       Ubuntu 22.04 Server                  null

# tvimtol9 cannot work with "A5000"

/*
# Attempts to use H100 did not work
machine_image_template = "t7vp562h"
machine_type="H100"
# t7vp562h  mliab-u22-241108-h100 :for H100:  "ML-in-a-Box 22.04 Template"
# not: "twnlo3zj", "t0nspur5", "t17d1a6i"
*/


# instance_disk_size_gb = 50
# in GB.
# NGC docker did not fit in 50 GB. Default is 100. I use 200, it's cheap.
# instance_disk_size_gb = 200
# This is NOT respected.
instance_disk_size_gb = 500


# removed:
# region_pdefault="Europe (AMS1)"


# region_parameter="East Coast (NY2)"
region_parameter="Europe (AMS1)"
# "East Coast (NY2)",	"West Coast (CA1)",	"Europe (AMS1)"

networkid_parameter = "nx86kbo3"

ssh_keypair_name_inputparam="paperspace_sosi_fromlinux"
# shoud match ... (?) in provisioning_scripts/terraform/common/per-provider/paperspace/setup_anew_ssh.bas
# but this file is deprecated, I think


# Ubuntu 20.04
# Region: UNKNOWN REGION --> ""AMS1 (Paperspace)"
# ML-in-a-Box
# Accelerators: Ampere A5000 (24 GB GPU memory)
