cmake_minimum_required(VERSION 3.17.0)

project(implicit)

enable_testing()

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(OpenMP)

include_directories(.)
set (CMAKE_CXX_STANDARD 11)
add_subdirectory(implicit)
