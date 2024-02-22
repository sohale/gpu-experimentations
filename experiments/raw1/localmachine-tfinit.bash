
# local machine or workstation

# For installtion, see: experiments/raw1/localmachine-installation.bash


# export TF_VAR_api_key="your_paperspace_api_key"

# terraform apply -var "api_key=mykey0010"
# terraform apply -var-file="secrets.tfvars"
#  which: api_key = "mykey0010"
# project_id = ""

terraform init -var-file="secrets.tfvars"  -var-file="tfconfig.tfvars"

# works
# next step, fails: "plan". See [4]
terraform plan -var-file="secrets.tfvars"  -var-file="tfconfig.tfvars"

# [4] https://blog.paperspace.com/introducing-paperspace-terraform-provider/

