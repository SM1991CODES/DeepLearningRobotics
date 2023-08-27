cmake_minimum_required(VERSION 3.10)
project(opencl_project)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} Main.c)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
