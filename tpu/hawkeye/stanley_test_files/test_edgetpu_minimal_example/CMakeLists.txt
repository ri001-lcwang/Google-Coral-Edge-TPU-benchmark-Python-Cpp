cmake_minimum_required(VERSION 3.11)
project(minimal-edgetpu-example)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "-Wall -pthread \
-g \
")
set(CMAKE_CXX_EXTRA_FLAGS -fext-numeric-literals 
-D_GNU_SOURCE 
-D_LARGEFILE64_SOURCE 
-D_FILE_OFFSET_BITS=64 
-D__STDC_CONSTANT_MACROS
)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++17 -lstdc++")


set(TF_ET_SRC_LIB "/home/pi/edgetpu-minimal-example/libedgetpu/direct/armv7a/libedgetpu.so.1.0")
set(TF_LITE_LIB "/home/pi/edgetpu-minimal-example/build/libtensorflow-lite.a")
include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(/home/pi/edgetpu-minimal-example/build/)
include_directories(/usr/include/)
include_directories(/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/)
include_directories(/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/absl/)
include_directories(/home/pi/edgetpu-minimal-example/include/)
include_directories(/home/pi/edgetpu-minimal-example/libedgetpu/)
include_directories(/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/flatbuffers/include/)

link_directories(${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

add_library(model_utils
  /home/pi/edgetpu-minimal-example/src/model_utils.cc
  /home/pi/edgetpu-minimal-example/include/model_utils.h)
target_compile_options(model_utils PRIVATE ${CMAKE_CXX_EXTRA_FLAGS} -O3)
target_link_libraries(model_utils ${TF_LITE_LIB})

add_library(misc
  sub_misc_c0.cpp sub_misc_c0.h
  parameter_set.h
  sub_librosa_c0.cpp sub_librosa_c0.h
  sub_ulib_realtime_pso_c0.cpp sub_ulib_realtime_pso_c0.h)
target_compile_options(misc PRIVATE -std=gnu++17 ${CMAKE_CXX_EXTRA_FLAGS} -O3)
target_link_libraries(misc lapacke fftw3 rt dl m asound)

add_library(thread_function
  thread_function.cpp thread_function.h
  parameter_set.h
)
target_compile_options(thread_function PRIVATE ${CMAKE_CXX_EXTRA_FLAGS})
target_link_libraries(thread_function misc model_utils ${TF_LITE_LIB} ${TF_ET_SRC_LIB}
)

add_executable(demo
  demo.cpp demo.h
  parameter_set.h
  /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
  /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
)
target_compile_options(demo PRIVATE ${CMAKE_CXX_EXTRA_FLAGS} -O3)
#target_link_libraries(demo ${TF_LITE_LIB} ${TF_ET_SRC_LIB} 
#  lapacke fftw3 rt dl m flatbuffers asound thread_function misc model_utils
#)
target_link_libraries(demo 
  thread_function misc model_utils ${TF_LITE_LIB} ${TF_ET_SRC_LIB}
)



#[[
add_library(minimal
  minimal.cc
  minimal.h
)
target_compile_options(minimal PRIVATE ${CMAKE_CXX_EXTRA_FLAGS} -O3)
target_link_libraries(minimal ${TF_LITE_LIB})

add_executable(demo
  demo.cpp
  demo.h
  /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
  /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
)
target_compile_options(demo PRIVATE ${CMAKE_CXX_EXTRA_FLAGS} -O3)
target_link_libraries(demo misc minimal model_utils ${TF_LITE_LIB} ${TF_ET_SRC_LIB})
#add_dependencies(demo minimal model_utils)
]]
