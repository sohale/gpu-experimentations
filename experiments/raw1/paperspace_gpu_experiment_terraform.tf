

resource "paperspace_gpu_instance" "example" {
  project_id =  var.project_id
  # Other properties: region, machine type, etc.
}

variable "api_key" {}

provider "paperspace" {
  api_key = var.api_key
  # More configuration here
}

#  either a
# terraform.tfvars file,
# command-line arguments,
# or environment variables.

terraform {
  required_providers {
    paperspace = {
      source = "paperspace/paperspace"
    }
  }
}
