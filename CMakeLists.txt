cmake_minimum_required(VERSION 3.22)

project(backpropagation)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wno-unused-result)

find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)

include_directories(SYSTEM ${AMADILLO_INCLUDE_DIR})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
add_compile_definitions(SRC_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

FILE(GLOB_RECURSE cpp_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(backpropagation ${cpp_sources})
target_link_libraries(backpropagation OpenMP::OpenMP_CXX)
target_link_libraries(backpropagation ${ARMADILLO_LIBRARIES})


file(MAKE_DIRECTORY tmp)

