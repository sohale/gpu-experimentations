# PowerShell script to add cl.exe to PATH
# Run in Windows


# -------- Prepare --------
# 1. permissions
# 2. set path

# Determine whether to use 32-bit or 64-bit compiler. Defaulting to 64-bit.
$architecture = "x64"
# If the output is "True", it is a 64-bit system, and it is OK.

# [System.Environment]::Is64BitOperatingSystem
# Then:
# cd gpu-experimentations\experiments\6_cuda_rggbuff
# Set-ExecutionPolicy RemoteSigned

# Construct the path to the cl.exe based on the architecture
$clPath = "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Tools\MSVC\14.16.27023\bin\Host$architecture\$architecture"

# Check if the path exists
if (Test-Path -Path $clPath) {
    # Add the path to the system PATH environment variable
    $env:Path += ";$clPath"
    Write-Host "Path added to the environment variable: $clPath"
} else {
    Write-Host "The path does not exist: $clPath"
}

# -------- Build: --------

# nvcc -o myexec main.cpp .\rgbproc_kernel.cu

# compile kernel
nvcc -c mykernel.cu -o mykernel.obj

# compile host
#### cl -c main.cpp -o main.obj  # Replace g++ with your C++ compiler if different
#### cl /c main.cpp /Fo"main.obj"
nvcc  -x cu -c host_main.cpp -o host_main.obj

# link step
nvcc -o myexec   host_main.obj mykernel.obj

