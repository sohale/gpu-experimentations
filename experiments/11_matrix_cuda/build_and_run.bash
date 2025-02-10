# sudo apt install nvidia-cuda-toolkit

echo 'To run the terraform:

bash provisioning_scripts/terraform/common/localmachine/up-matmul-cuda-experiment.bash tfapply
'
# echo "which in turn"
# find provisioning_scripts/terraform/common/localmachine/upload_scripts_to_there.bash
# echo "which in turn"
# find provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/environment-box/local_manual__setup_at_creation.bash

nvcc -o matrix_mult main.cpp matrix_kernel.cu

echo "to run:"
echo "./matrix_mult"
