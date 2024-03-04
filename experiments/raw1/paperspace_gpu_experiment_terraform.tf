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

resource "paperspace_script" "my-startup-script-1" {
  name = "My Script"
  description = "a short description"
  script_text = <<EOF
#!/bin/bash
echo "Hello, World" > index.html
ufw allow 8080
nohup busybox httpd -f -p 8080 &
EOF
  is_enabled = true
  run_once = false
}

resource "paperspace_machine" "my-machine-1" {
  region = "East Coast (NY2)" // optional, defaults to provider region if not specified
  name = "Terraform Test"
  machine_type = "C1"
  size = 50
  billing_type = "hourly"
  assign_public_ip = false // optional, remove if you don't want a public ip assigned

  template_id = data.paperspace_template.my-template-1.id
  user_id = data.paperspace_user.my-user-1.id  // optional, remove to default
  team_id = data.paperspace_user.my-user-1.team_id
  script_id = paperspace_script.my-startup-script-1.id
  shutdown_timeout_in_hours = 1
  # live_forever = true # enable this to make the machine have no shutdown timeout
}
