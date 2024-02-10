
provider "paperspace" {
  api_key = "mykey0010"
  # Additional configuration here
}

resource "paperspace_gpu_instance" "example" {
  project_id = "id00011"
  # Define other properties like region, machine type, etc.
}

variable "api_key" {}

provider "paperspace" {
  api_key = var.api_key
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
