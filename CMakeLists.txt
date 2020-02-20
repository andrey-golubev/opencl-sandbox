cmake_minimum_required(VERSION 2.8)
project(ocl)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# experimental warning arguments (e.g. deprecated-declarations requried for OpenCL 1.2 functionality)
set(WARNARGS_EXPERIMENTAL "-Wno-deprecated-declarations -Wno-unused-but-set-variable -Wno-terminate")
# warning arguments
set(WARNARGS "-Wall -Werror -Wno-unused-function -Wno-ignored-attributes ${WARNARGS_EXPERIMENTAL}")

#ASan
option(WITH_ASAN "Enable ASan build flags" OFF)
if (${WITH_ASAN})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fsanitize=undefined")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libsan")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libasan")
  endif()
endif (${WITH_ASAN})

option(WITH_THREADING "Enable threaded execution" OFF)
if (${WITH_THREADING})
  add_compile_definitions(OMP_THREADING)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif (${WITH_THREADING})

# rules for debug and release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0 ${WARNARGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 ${WARNARGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -g -O2 ${WARNARGS}")

# include root folder
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# subfolders
add_subdirectory(img_processor)
add_subdirectory(eltwise_mat_add)
add_subdirectory(rgb2gray_blur)
add_subdirectory(stereo_disparity)
