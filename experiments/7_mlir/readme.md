# The MLIR experiment
aka "experiment 7".

An experiment in (provisionally) series of MLIR experiments. This itself is not the first experiment.

* created a successful docker container for MLIR development verison
    * prebuilt, but can be built again.
    * Includes development tools
    * Fast: Layers are desigend to fast rebuild, etc
    * Build of Dockerfile is fast (maximizing layers)
    * Mitigated the long time for clone & for build of LLVM
        * And ability to benefit from past builds (incremetnal "docker-commit" s)

* Learnings are documented in: https://github.com/sohale/cs-glossaries/blob/master/compilers/MLIR-plunge.md

### sub-projects
Sub-projects in MLIR-7 experiment:
* Project 1: Building MLIR, Docker for dev & use, environment to create dialects, etc
   * Use of MLIR
   * Dev of MLIR
* Project 2: A TableGen
    * Prepares for geration of a dialect
* Project 3: A dialect (provisional)
* Project 4: MLIR code generation (provisional)
