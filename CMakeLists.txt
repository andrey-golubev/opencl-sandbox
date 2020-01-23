cmake_minimum_required(VERSION 2.8)
project(ocl)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Werror -Wno-unused-function")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unused-function")

# subfolders
add_subdirectory(img_processor)
add_subdirectory(eltwise_mat_sum)
