cmake_minimum_required(VERSION 3.17)

project(TFLite)

set(CMAKE_CXX_STANDARD 14)

# OpenCV Integration
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)
add_executable(${PROJECT_NAME} tflite.cpp)


add_library(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite.so)
target_link_libraries(${PROJECT_NAME} tensorflowlite ${OpenCV_LIBS})