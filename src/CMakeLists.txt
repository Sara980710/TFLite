cmake_minimum_required(VERSION 3.16)

project(TFLite)

set(CMAKE_CXX_STANDARD 14)

# OpenCV Integration
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Using OpenCV ${OpenCV_VERSION}")

find_package(OpenGL REQUIRED COMPONENTS EGL GLX) # OpenGL
message(STATUS "Using OpenGL ${OpenGL_VERSION}")

# TensorFlow Lite Integration
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)
set(TFLITE_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/")

# Set names
set(YOLO TFLiteYolov5)
set(CLASS TFLiteClassification)
set(TIME TFLiteTimeit)
set(MEMORY TFLiteMemory)

# Create executable files
add_executable(${YOLO} tflite_yolo.cpp Detector.cpp Detector.h)
add_executable(${CLASS} tflite_classification.cpp Detector.cpp Detector.h)
add_executable(${TIME} tflite_timeit.cpp Detector.cpp Detector.h)
add_executable(${MEMORY} tflite_mem.cpp Detector.cpp Detector.h)

# TFLite GPU libraries
foreach(_target
    libtensorflowlite
    libtensorflowlite_gpu_delegate
    libtensorflowlite_gpu_gl
    libtensorflowlite_xnnpack)

  add_library(${_target} SHARED IMPORTED)
  set_target_properties(${_target} PROPERTIES IMPORTED_LOCATION ${TFLITE_LIBS_DIR}/${_target}.so)
  target_link_libraries(${_target} INTERFACE OpenGL::GLX OpenGL::EGL)
  list(APPEND TFLITE_LIBS ${_target})
endforeach()


# Link libraries
target_link_libraries(${YOLO} ${TFLITE_LIBS} ${OpenCV_LIBS})
target_link_libraries(${CLASS} ${TFLITE_LIBS} ${OpenCV_LIBS})
target_link_libraries(${TIME} ${TFLITE_LIBS} ${OpenCV_LIBS})
target_link_libraries(${MEMORY} ${TFLITE_LIBS} ${OpenCV_LIBS})

