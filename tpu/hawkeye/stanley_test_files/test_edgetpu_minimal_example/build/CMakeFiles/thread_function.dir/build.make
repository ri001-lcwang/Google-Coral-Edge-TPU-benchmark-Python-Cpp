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
include CMakeFiles/thread_function.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/thread_function.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/thread_function.dir/flags.make

CMakeFiles/thread_function.dir/thread_function.cpp.o: CMakeFiles/thread_function.dir/flags.make
CMakeFiles/thread_function.dir/thread_function.cpp.o: ../thread_function.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/thread_function.dir/thread_function.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/thread_function.dir/thread_function.cpp.o -c /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/thread_function.cpp

CMakeFiles/thread_function.dir/thread_function.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/thread_function.dir/thread_function.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/thread_function.cpp > CMakeFiles/thread_function.dir/thread_function.cpp.i

CMakeFiles/thread_function.dir/thread_function.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/thread_function.dir/thread_function.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/thread_function.cpp -o CMakeFiles/thread_function.dir/thread_function.cpp.s

# Object files for target thread_function
thread_function_OBJECTS = \
"CMakeFiles/thread_function.dir/thread_function.cpp.o"

# External object files for target thread_function
thread_function_EXTERNAL_OBJECTS =

libthread_function.a: CMakeFiles/thread_function.dir/thread_function.cpp.o
libthread_function.a: CMakeFiles/thread_function.dir/build.make
libthread_function.a: CMakeFiles/thread_function.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libthread_function.a"
	$(CMAKE_COMMAND) -P CMakeFiles/thread_function.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/thread_function.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/thread_function.dir/build: libthread_function.a

.PHONY : CMakeFiles/thread_function.dir/build

CMakeFiles/thread_function.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/thread_function.dir/cmake_clean.cmake
.PHONY : CMakeFiles/thread_function.dir/clean

CMakeFiles/thread_function.dir/depend:
	cd /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build /home/pi/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build/CMakeFiles/thread_function.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/thread_function.dir/depend

