# Declare variables before using the,

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

/*
Error: Invalid resource type: │ The provider paperspace/paperspace does not support resource type "paperspace_gpu_instance"

resource "paperspace_gpu_instance" "example" {
  project_id =  var.project_id
  # Other properties: region, machine type, etc.
}
*/

variable "api_key" {
  description = "My API key"
  type        = string
}

provider "paperspace" {
  region = "East Coast (NY2)"
  api_key = var.api_key
  # More configuration here
}

#  either a
# terraform.tfvars file,
# command-line arguments,
# or environment variables.

# Why??
data "paperspace_template" "my-template-1" {
  id = "t04azgph" // this is one of the Ubuntu Server 18.04 templates
}
data "paperspace_user" "my-user-1" {
  email = "sohale@gmail.com"
  team_id = "t10oyrfnaf"
}

output "api_key" {
  value = var.api_key
  sensitive = true
}

terraform {
  required_providers {
    paperspace = {
      source = "paperspace/paperspace"
    }
  }
}
