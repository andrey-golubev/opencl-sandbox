cmake_minimum_required(VERSION 2.8)
project(eltwise_mat_add)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} opencv_core OpenCL)
