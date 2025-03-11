cmake_minimum_required(VERSION 3.10)
project(nvidia_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} 
    src/main_nvidia.cpp
)

find_package(CUDA REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

