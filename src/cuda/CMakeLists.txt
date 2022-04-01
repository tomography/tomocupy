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

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(cfunc_fourierrec.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_fourierrec LANGUAGE python SOURCES cfunc_fourierrec.i cfunc_fourierrec.cu)
swig_link_libraries(cfunc_fourierrec cudart cufft)
set_target_properties(cfunc_fourierrec PROPERTIES CUDA_ARCHITECTURES "60;72;75")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_fourierrec.py" DESTINATION tomocupyfp16_cli)
install(TARGETS cfunc_fourierrec LIBRARY DESTINATION tomocupyfp16_cli)


set_source_files_properties(cfunc_fourierrecfp16.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_fourierrecfp16 LANGUAGE python SOURCES cfunc_fourierrecfp16.i cfunc_fourierrec.cu)
swig_link_libraries(cfunc_fourierrecfp16 cudart cufft)
set_target_properties(cfunc_fourierrecfp16 PROPERTIES CUDA_ARCHITECTURES "60;72;75")
target_compile_definitions(cfunc_fourierrecfp16 PRIVATE HALF)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_fourierrecfp16.py" DESTINATION tomocupyfp16_cli)
install(TARGETS cfunc_fourierrecfp16 LIBRARY DESTINATION tomocupyfp16_cli)

