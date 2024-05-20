cmake_minimum_required(VERSION 3.5)
project(torchtest)

set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

#
# MNIST
#

# MSGD
add_executable(msgd_mnist msgd.cpp convnet.cpp)
target_link_libraries(msgd_mnist ${TORCH_LIBRARIES} ${MPI_LIBRARIES})
set_property(TARGET msgd_mnist PROPERTY CXX_STANDARD 17)

# EAMSGD
add_executable(eamsgd_mnist eamsgd.cpp convnet.cpp)
target_link_libraries(eamsgd_mnist ${TORCH_LIBRARIES} ${MPI_LIBRARIES})
set_property(TARGET eamsgd_mnist PROPERTY CXX_STANDARD 17)

# EASGD
add_executable(easgd_mnist easgd.cpp convnet.cpp)
target_link_libraries(easgd_mnist ${TORCH_LIBRARIES} ${MPI_LIBRARIES})
set_property(TARGET easgd_mnist PROPERTY CXX_STANDARD 17)

#
# CIFAR-10
#
# MSGD
add_executable(msgd_cifar msgd_cifar.cpp convnet_cifar.cpp cifar10.cpp)
target_link_libraries(msgd_cifar ${TORCH_LIBRARIES} ${MPI_LIBRARIES})
set_property(TARGET msgd_cifar PROPERTY CXX_STANDARD 17)

# EAMSGD
add_executable(eamsgd_cifar eamsgd_cifar.cpp convnet_cifar.cpp cifar10.cpp)
target_link_libraries(eamsgd_cifar ${TORCH_LIBRARIES} ${MPI_LIBRARIES})
set_property(TARGET eamsgd_cifar PROPERTY CXX_STANDARD 17)

# EASGD
add_executable(easgd_cifar easgd_cifar.cpp convnet_cifar.cpp cifar10.cpp)
target_link_libraries(easgd_cifar ${TORCH_LIBRARIES} ${MPI_LIBRARIES})
set_property(TARGET easgd_cifar PROPERTY CXX_STANDARD 17)

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