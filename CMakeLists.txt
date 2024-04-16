cmake_minimum_required(VERSION 3.5)
project(torchtest)

set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)

find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torchtest src/torchtest.cpp)

target_link_libraries(torchtest ${TORCH_LIBRARIES})

set_property(TARGET torchtest PROPERTY CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED_ON)

target_link_libraries(torchtest ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(dist-mnist PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(dist-mnist PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()