cmake_minimum_required(VERSION 3.10)
project(MyMLIRProject)


# Set the path to your LLVM and MLIR installations
# set(LLVM_DIR /mlir/llvm-project//lib/cmake/llvm)
# set(MLIR_DIR /mlir/llvm-project/lib/cmake/mlir)

# set(LLVM_DIR /mlir/llvm-project/build/lib/cmake/llvm)
# set(MLIR_DIR /mlir/llvm-project/build/lib/cmake/mlir)

set(LLVM_DIR "/mlir/llvm-project/build/lib/cmake/llvm")
set(MLIR_DIR "/mlir/llvm-project/build/lib/cmake/mlir")


# Find LLVM and MLIR packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# include_directories(${LLVM_INCLUDE_DIRS})
# include_directories(${MLIR_INCLUDE_DIRS})


# Include LLVM and MLIR directories
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddMLIR)
include(TableGen)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

set(CMAKE_CXX_STANDARD 14)

# set(LLVM_TARGET_DEFINITIONS include/my_mlir_project/MyOps.td)

# tut1
# set(LLVM_TARGET_DEFINITIONS myops.td)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/myops.td)
mlir_tablegen(myops.h.inc -gen-op-decls)
mlir_tablegen(myops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MyOpsIncGen)

add_library(MyMLIRProject
  mydialect.cpp   # lib/MyDialect.cpp
  myops.cpp    # lib/MyOps.cpp
  # DEPENDS
  # MyOpsIncGen
)

# Add dependency for TableGen generated files
add_dependencies(MyMLIRProject MyOpsIncGen)

target_link_libraries(MyMLIRProject PRIVATE MLIRIR)

# MyMLIRProject <-> TableGenExample
