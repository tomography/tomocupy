enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# SWIG requires the Python header
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
# TODO: Is NumPy required?
#find_package(NumPy REQUIRED)
#include_directories(${NumPy_INCLUDE_DIRS})



#add_definitions(-DHALF)


# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(cfunc_lprec.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_lprec LANGUAGE python SOURCES cfunc_lprec.i cfunc_lprec.cu)
swig_link_libraries(cfunc_lprec cudart cufft)
set_target_properties(cfunc_lprec PROPERTIES CUDA_ARCHITECTURES "60;72")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_lprec.py" DESTINATION h5gpurec)
install(TARGETS cfunc_lprec LIBRARY DESTINATION h5gpurec)

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(cfunc_fourierrec.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_fourierrec LANGUAGE python SOURCES cfunc_fourierrec.i cfunc_fourierrec.cu)
swig_link_libraries(cfunc_fourierrec cudart cufft)
set_target_properties(cfunc_fourierrec PROPERTIES CUDA_ARCHITECTURES "60;72")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_fourierrec.py" DESTINATION h5gpurec)
install(TARGETS cfunc_fourierrec LIBRARY DESTINATION h5gpurec)
