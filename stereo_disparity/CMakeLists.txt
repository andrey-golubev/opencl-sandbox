cmake_minimum_required(VERSION 2.8)
project(stereo_disparity)

set(LINK_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_highgui OpenCL pthread)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})

add_executable(${PROJECT_NAME}_test test_acc.cpp)
target_link_libraries(${PROJECT_NAME}_test ${LINK_LIBS})

add_executable(${PROJECT_NAME}_perf test_perf.cpp)
target_link_libraries(${PROJECT_NAME}_perf ${LINK_LIBS})
