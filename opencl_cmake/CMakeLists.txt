cmake_minimum_required(VERSION 3.10)
project(opencl_project)

add_executable(${PROJECT_NAME} Main.c)

target_include_directories(${PROJECT_NAME} PRIVATE /usr/local/cuda/include)

target_link_directories(${PROJECT_NAME} PRIVATE /usr/local/cuda/lib64)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL)
