# gpu-experimentations
gpu/cuda/nvidia experimentations for cloud of GPU

Experiments:


1. 🧊 Terraform: `raw1`: Simple Terraform exprimentation. Later devloped into [https://github.com/sohale/gpu-experimentations/tree/main/provisioning_scripts/terraform]

2. 🧊 LLVM: `raw-llvm` : Coding LLVM hard-coded hands-on home-made LLVM code

3. 🧊 TVM: `3tvm` : TVM (Framework/DSL for Neural Networks Inference). (For TVM open-source tickets)

4. 🧊 Lean4: `4leannet1` (moved to 5) Early Lean4 experiments

4. 🧊 Triton: `4triton` (For OpenAI Triton open-source tickets)

5. 🧊 Lean4: `leannn5` Lean4 experiments

6. 🧊 CUDA: `6_cuda_rggbuff` Simple CUDA code for RGBA buffer

7. 🧊 MLIR: `7_mlir` MLIR experiment 1: Full MLIR build, build scripts, My own Docker build (Dockerfile) for containerised MLIR development (For MLIR open-source tickets)

8. 🧊 MLIR:`8_mlir_nn`: MLIR experiment 2: Neural network (cancelled)

9. 🧊 MLIR:`9_mlir_neo_refactor`: MLIR experiment 3  Neural network (with better build and container), as support for a compiler project. See [https://github.com/sohale/gpu-experimentations/tree/main/provisioning_scripts/mlir_env]. Also LLVM debugging using `lldb` (Clang toolchain).

10. 🧊 **PTX**: `10_mcmc_ptx`: MCMC using PTX (direct hard-coded NVidia's assembly language, on top of).
    * Low-level “Parallel-Thread Execution ISA Version 8.3” (almost architecture-independent, using "as-if virtual machine")
    * ( see [PTX (pdf)](https://docs.nvidia.com/cuda/pdf/PTX_Writers_Guide_To_Interoperability.pdf) and [PTX (html)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
    * `PTX` is itself on top of `SASS` (propriatory): [SASS' .yacc](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/cuobjdump_to_ptxplus/sass.y) and [SASS' .lex](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/cuobjdump_to_ptxplus/sass.l)
    * `ptxas`, ``
    * Also see [cuda_api.h](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/libcuda/cuda_api.h) and [cuda_runtime_api.cc](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/libcuda/cuda_runtime_api.cc) , [.lex file ptx.l](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/cuobjdump_to_ptxplus/ptx.l) on `gpgpusim`
    * PTX Op Codes: [opcodes.def](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/src/cuda-sim/opcodes.def)
    * Cool from GPGPUSIM: [gpgpu_context.h](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/libcuda/gpgpu_context.h). They even have OpenCL runtime API: [opencl_runtime_api.cc](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/libopencl/opencl_runtime_api.cc)
    * CUDA-level:  [cuda_runtime_api.cc](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/libcuda/cuda_runtime_api.cc) for CUDA-level and [instructions.cc](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/src/cuda-sim/instructions.cc)


12. 🧊 CUDA: `11_matrix_cuda`: Advanced CUDA optimisation techniques + profiling: for Matrix Multiplicaiton

13. 🧊 FPGA: `12_fpga_aws`: FPGA on cloud using AWS's F2, utilixiing Xilinx hardware and AmaranthDHL (open-source hardware HDL) (as part of heterogeneous computing)

14. 🧊 CUDA: `13_cuda_sharedmem`: Advanced CUDA+PTX optimisation techniques + profiling ( for experimentation with CUDA / CC Architectures )



