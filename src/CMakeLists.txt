cmake_minimum_required(VERSION 3.17)

project(TFLite)

set(CMAKE_CXX_STANDARD 14)

# OpenCV Integration
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)

# Set names
set(YOLO TFLiteYolov5)
set(CLASS TFLiteClassification)
set(TIME TFLiteTimeit)
set(MEMORY TFLiteMemory)

# Create executable files
add_executable(${YOLO} tflite_yolo.cpp Detector.cpp Detector.h)
add_executable(${CLASS} tflite_classification.cpp Detector.cpp Detector.h)
add_executable(${TIME} tflite_timeit.cpp)
add_executable(${MEMORY} tflite_mem.cpp)


add_library(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite.so)
#set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite_c.so)
target_link_libraries(${YOLO} tensorflowlite ${OpenCV_LIBS})
target_link_libraries(${CLASS} tensorflowlite ${OpenCV_LIBS})
target_link_libraries(${TIME} tensorflowlite ${OpenCV_LIBS})
target_link_libraries(${MEMORY} tensorflowlite ${OpenCV_LIBS})