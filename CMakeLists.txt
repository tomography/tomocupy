# 3.11.0 is from scikit-build documentation
cmake_minimum_required(VERSION 3.11.0)

project(tomocg LANGUAGES CXX)

# ---- DEFINE BUILD OPTIONS HERE ---- #
set(EXTENSION_WRAPPER swig
    CACHE STRING "Method used for wrapping Python C/CXX extensions.")
set_property(CACHE EXTENSION_WRAPPER PROPERTY STRINGS swig pybind11)
# ---- END BUILD OPTION DELARATIONS ---- #

# Add 'src' to install path so compiled modules go to the right place
file(TO_NATIVE_PATH "${CMAKE_INSTALL_PREFIX}/src" CMAKE_INSTALL_PREFIX)

add_subdirectory(src)