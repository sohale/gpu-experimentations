provider "paperspace" {
  # Configure Paperspace provider settings
}

resource "paperspace_instance" "gpu_instance" {
  # Define instance settings (e.g., GPU type, instance type, etc.)
}

resource "null_resource" "clone_repo" {
  provisioner "local-exec" {
    # Execute script to clone GitHub repository into Docker container
    command = "docker exec <container_id> git clone https://github.com/sohale/gpu-experimentations /path/to/raw1"
  }
  depends_on = [paperspace_instance.gpu_instance]
}

resource "null_resource" "compile_cuda_code" {
  provisioner "local-exec" {
    # Execute script to compile CUDA code
    command = "docker exec <container_id> /path/to/compile_script.sh"
  }
  depends_on = [null_resource.clone_repo]
}

# Logic to shutdown and destroy Paperspace instance after compilation
