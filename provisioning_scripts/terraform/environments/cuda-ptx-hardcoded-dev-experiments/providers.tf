provider "paperspace" {
  api_key = var.api_key

  # This is the "default" (a provider-level default), when `var.region_parameter` (?) is not specified (in the `main.tf`).
  region  = var.region_pdefault

}
# "paperspace_machine" is the resource type provided by the Paperspace provider
# https://github.com/Paperspace/terraform-provider-paperspace/blob/16187c6a315724d8e6a5be74222343f290b029cf/pkg/provider/provider.go#L42

# resources provided:
#    paperspace_machine
#    paperspace_script
#    paperspace_network

# datasources provided:
#    paperspace_user
#    paperspace_network

# Schema (inputs?):
#    api_key
#    api_host
#    region
# env inputs: (One for each schema above)
#   PAPERSPACE_API_KEY

# paperspace_network is bothernames frot wo things resource and datasource

#source
# api_host: envDefaultFuncAllowMissingDefault("PAPERSPACE_API_HOST", "https://api.paperspace.io"),

terraform {
  required_providers {
    paperspace = {
      source = "paperspace/paperspace"
    }
  }
}

# Paperspace status page
# https://status.paperspace.com/


# cli
# https://docs.digitalocean.com/reference/paperspace/pspace/install/

/*

ONCE \
curl -fsSL https://paperspace.com/install.sh | sh

pspace machine list
pspace machine delete
pspace secret list # ?
pspace os-template list  # nice
*/

# see variables.tf for machine type code (id)s
