# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/demo.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/demo.cpp.o: ../demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo.dir/demo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/demo.cpp.o -c /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/demo.cpp

CMakeFiles/demo.dir/demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/demo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/demo.cpp > CMakeFiles/demo.dir/demo.cpp.i

CMakeFiles/demo.dir/demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/demo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/demo.cpp -o CMakeFiles/demo.dir/demo.cpp.s

CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.o: /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.o   -c /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c

CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c > CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.i

CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c -o CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.s

CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.o: /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.o -c /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc

CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc > CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.i

CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc -o CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.s

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/demo.cpp.o" \
"CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.o" \
"CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

../out/demo: CMakeFiles/demo.dir/demo.cpp.o
../out/demo: CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c.o
../out/demo: CMakeFiles/demo.dir/home/pi/edgetpu-minimal-example/build/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc.o
../out/demo: CMakeFiles/demo.dir/build.make
../out/demo: libthread_function.a
../out/demo: libmisc.a
../out/demo: libmodel_utils.a
../out/demo: /home/pi/edgetpu-minimal-example/build/libtensorflow-lite.a
../out/demo: /home/pi/edgetpu-minimal-example/libedgetpu/direct/armv7a/libedgetpu.so.1.0
../out/demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../out/demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: ../out/demo

.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo.dir/depend

