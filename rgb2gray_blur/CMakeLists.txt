cmake_minimum_required(VERSION 2.8)
project(rgb2gray_blur)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgcodecs opencv_imgproc OpenCL)

add_executable(${PROJECT_NAME}_test test_acc.cpp)
target_link_libraries(${PROJECT_NAME}_test opencv_core opencv_imgcodecs opencv_imgproc OpenCL)
