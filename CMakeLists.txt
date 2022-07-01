cmake_minimum_required(VERSION 3.22)

project(backprogation VERSION 3.14)

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

add_executable(backprogation ${cpp_sources})
target_link_libraries(backprogation OpenMP::OpenMP_CXX)
target_link_libraries(backprogation ${ARMADILLO_LIBRARIES})


file(MAKE_DIRECTORY tmp)

