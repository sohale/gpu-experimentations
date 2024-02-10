
# local machine or workstation

# (deprecated) Installing on MacOS [1]

# brew install terraform
# terraform -v
# update ( based on [2] )
# But it's deprecated
# brew uninstall terraform
# brew autoremove

# new version based on [3]
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

terraform -install-autocomplete
# terraform -uninstall-autocomplete

brew update  # brew itself
brew upgrade hashicorp/tap/terraform

# export TF_VAR_api_key="your_paperspace_api_key"

# terraform apply -var "api_key=mykey0010"
# terraform apply -var-file="secrets.tfvars"
#  which: api_key = "mykey0010"


# [1] Based on: https://docs.digitalocean.com/reference/terraform/getting-started/
# [2] Upgrade: https://www.terraform.io/downloads.html
# [3] New instructions: https://developer.hashicorp.com/terraform/install

