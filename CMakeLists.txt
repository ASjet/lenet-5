cmake_minimum_required(VERSION 3.5)

project(HandWrittenDightRecognition)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	/usr/local/include/eigen3
	/usr/local/include/opencv4
)

add_executable(
	test
	src/eigentest.cpp
)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-O0 -g)
