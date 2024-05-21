cmake_minimum_required(VERSION 3.10)
project(MyMLIRProject)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

include(TableGen)

set(LLVM_TARGET_DEFINITIONS include/my_mlir_project/MyOps.td)
mlir_tablegen(MyOps.h.inc -gen-op-decls)
mlir_tablegen(MyOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MyOpsIncGen)

add_library(MyMLIRProject
  lib/MyDialect.cpp
  lib/MyOps.cpp
  DEPENDS
  MyOpsIncGen
)

target_link_libraries(MyMLIRProject PRIVATE MLIRIR)


