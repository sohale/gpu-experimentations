# Team:
# > Paperspace
# https://console.paperspace.com/entobrain/settings/profile
# ( CORE? , Gradiaent, Paperspace)
# name=entobrain
# U=entobrain
# ID=tqpio01i96

# team_u = "entobrain"
team_id="tqpio01i96"
project_id="p0k609uoxdl"
project_name = "entobrain-dl1"

apikey_secret_name="entobrain_apikey"

input_public_ip="9.9.9.9"

instance_name = "Physical A-5000 (by Terraform)"


# machine_image_template = "twnlo3zj"  # (A5000?)
# machine_image_template = "t04azgph"  # Ubuntu Server 18.04
# machine_type="A5000"
# machine_type= "C1"

machine_image_template = "twnlo3zj"
machine_type="A5000"

region_pdefault="Europe (AMS1)"
region_parameter="Europe (AMS1)"

ssh_keypair_name_inputparam="paperspace_sosi_fromlinux"
# shoud match ... (?) in provisioning_scripts/terraform/common/per-provider/paperspace/setup_anew_ssh.bas
# but this file is deprecated, I think


# Ubuntu 20.04
# Region: UNKNOWN REGION --> ""AMS1 (Paperspace)"
# ML-in-a-Box
# Accelerators: Ampere A5000 (24 GB GPU memory)
