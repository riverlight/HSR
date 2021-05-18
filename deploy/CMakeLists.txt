cmake_minimum_required(VERSION 2.8.9)
project (hsisr)

message ("$ENV{LIBTORCH_ROOT} ")
message ("$ENV{OPENCV4_LIB}")


set (CMAKE_BUILD_TYPE release)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS}  -O3 -Wall -DNDEBUG")

include_directories( $ENV{LIBTORCH_ROOT}/include/torch/csrc/api/include $ENV{LIBTORCH_ROOT}/include $ENV{OPENCV4_INCLUDE})

set (SRC hsisr.cpp )

set ( PROJECT_LINK_LIBS $ENV{LIBTORCH_ROOT}/lib/*.lib opencv_world411.lib )
link_directories( $ENV{LIBTORCH_ROOT}/lib $ENV{OPENCV4_LIB})

add_executable(hsisr ${SRC})
target_link_libraries(hsisr ${PROJECT_LINK_LIBS} )

