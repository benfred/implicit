cmake_minimum_required(VERSION 3.21...4.0)

project(implicit)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)
include(UseCython)

enable_testing()

find_package(OpenMP)

include_directories(.)
set (CMAKE_CXX_STANDARD 17)
add_subdirectory(implicit)
