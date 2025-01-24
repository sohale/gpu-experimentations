#!/bin/bash

set -ex

export PROGNAME=loop_vector_computation

# Assemble the LLVM IR Code
# llvm-as loop_vector_computation.llvm
# llvm-as loop_vector_computation.llvm  -o loop_vector_computation.bc
llvm-as $PROGNAME.ll  -o $PROGNAME.bc

# Generate Native Assembly
# llc loop_vector_computation.bc -o loop_vector_computation.s
# nit used, but to have an updated *.s file
llc $PROGNAME.bc -o $PROGNAME.s
# no -g
#
# obj: object file (machine code)
# asm: (assembly code)
# null (no output)
llc -filetype=obj -O0 $PROGNAME.bc -o $PROGNAME.o


# Compile to Executable
# clang loop_vector_computation.s -o loop_vector_computation.executable -lm
# clang $PROGNAME.s -o $PROGNAME.executable -lm
clang -g $PROGNAME.o -o $PROGNAME

# ./$PROGNAME.executable
# Segmentation fault (core dumped)
# Debug using LLDB
lldb ./$PROGNAME
