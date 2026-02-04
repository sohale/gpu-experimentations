cmake_minimum_required(VERSION 3.20)

# Project setup
project(cpp_test1 LANGUAGES CXX)

# Default to the newest widely supported standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Executable target
add_executable(cpp_test1 cpp_test1.cpp)

# Optional (cmake cli arg: `-DUSE_CPP26=ON` ): attempt C++26 (experimental) if supported
option(USE_CPP26 "Attempt to compile with experimental C++26 (-std=c++2c)" OFF)
if(USE_CPP26)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    include(CheckCXXCompilerFlag)

    # -std=c++2c === -std=c++26
    # -std=c++2b === -std=c++23
    # -std=c++20
    # -std=c++17

    # C++ standard compiler-flag: Going for C++26
    check_cxx_compiler_flag("-std=c++2c" HAS_CPP2C)
    if(HAS_CPP2C)
      target_compile_options(cpp_test1 PRIVATE "-std=c++2c")
      message(STATUS "C++ version: Building with experimental C++26 via -std=c++2c (no fallback)")
    else()
      # Fallback to pre-release flag often used for the next standard
      check_cxx_compiler_flag("-std=c++2b" HAS_CPP2B)
      if(HAS_CPP2B)
        target_compile_options(cpp_test1 PRIVATE "-std=c++2b")
        message(STATUS "C++ version: Since -std=c++2c was not supported,")
        message(STATUS "C++ version: Building with pre-release next-standard via -std=c++2b instad (fallback 1).")
      else()
        message(WARNING "Neither -std=c++2c nor -std=c++2b supported. Using C++23.")
      endif()
    endif()

  elseif(MSVC)
    target_compile_options(cpp_test1 PRIVATE "/std:c++latest")
    message(STATUS "C++ version: Building with MSVC /std:c++latest")
  else()
    message(WARNING "Unknown compiler; cannot enable C++26. Using C++23.")
  endif()
endif()
