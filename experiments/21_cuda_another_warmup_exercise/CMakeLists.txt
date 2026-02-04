cmake_minimum_required(VERSION 3.20)

# Project setup
project(cpp_test1 LANGUAGES CXX)

# Default to the newest widely supported standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Executable target
add_executable(cpp_test1 cpp_test1.cpp)

# Optional: attempt C++26 (experimental) if supported
option(USE_CPP26 "Attempt to compile with experimental C++26 (-std=c++2c)" OFF)
if(USE_CPP26)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-std=c++2c" HAS_CPP2C)
    if(HAS_CPP2C)
      target_compile_options(cpp_test1 PRIVATE "-std=c++2c")
      message(STATUS "Building with experimental C++26 via -std=c++2c")
    else()
      message(WARNING "C++26 flag -std=c++2c not supported by this compiler. Using C++23.")
    endif()
  elseif(MSVC)
    target_compile_options(cpp_test1 PRIVATE "/std:c++latest")
    message(STATUS "Building with MSVC /std:c++latest")
  else()
    message(WARNING "Unknown compiler; cannot enable C++26. Using C++23.")
  endif()
endif()
