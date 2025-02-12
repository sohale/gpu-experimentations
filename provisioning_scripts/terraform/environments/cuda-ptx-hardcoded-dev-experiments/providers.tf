provider "paperspace" {

  api_key = var.api_key

  # region =
  # The following may be worng: this may be "Provider/agent/server's region"
  # This is the "default" (a provider-level default), when `var.region_parameter` (?) is not specified (in the `main.tf`).
  # It is meaningless to parametrise it (`region  = var.region_pdefault` )
  # It is also "provider-level": ince per provider
  # Also it is about the agent that does that?
  # region = "Europe (AMS1)"
  #    "East Coast (NY2)",	"West Coast (CA1)",	"Europe (AMS1)"
  # If default
  # region = "Europe (AMS1)"
  # If it's provider-agent's region:
  region = var.region_parameter

  # arguments:
  # api_key: $PAPERSPACE_API_KEY
  # api_host: $PAPERSPACE_API_HOST or "https://api.paperspace.io"
  # region: $PAPERSPACE_REGION (can be "")

  # See `--api-key`, `--api-url` in `pspace` CLI.
}

#

/*
A "provider" of type "paperspace", enables:   https://github.com/Paperspace/terraform-provider-paperspace/blob/master/pkg/provider/resource_machine.go


Binds to: (as parameter, input, arguments, properties)
    datasource: (data)
        user (email, firstname, lastname, dt_created, team_id)
        template
        network (name, region, dt_created, network, netmask, team_id)
        job_storage (team_id: int, region)

Allocates:
    resources:
        machine
        network   (database id)
        script (run_once)
        script (non-run_once)
        autoscaling_group (name,machine_type,template_id,network_id,startup_script_id)
*/

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
