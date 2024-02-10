
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



# [1] Based on: https://docs.digitalocean.com/reference/terraform/getting-started/
# [2] Upgrade: https://www.terraform.io/downloads.html
# [3] New instructions: https://developer.hashicorp.com/terraform/install

