cmake_minimum_required(VERSION 3.5)
project(torchtest)

set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# MSGD
add_executable(msgd msgd_cifar.cpp convnet_cifar.cpp cifar10.cpp)
target_link_libraries(msgd ${TORCH_LIBRARIES} ${MPI_LIBRARIES})
set_property(TARGET msgd PROPERTY CXX_STANDARD 17)

# EAMSGD
add_executable(eamsgd eamsgd_cifar.cpp convnet_cifar.cpp cifar10.cpp)
target_link_libraries(eamsgd ${TORCH_LIBRARIES} ${MPI_LIBRARIES})
set_property(TARGET eamsgd PROPERTY CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED_ON)

if(MPI_COMPILE_FLAGS)
  set_target_properties(main PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(main PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()