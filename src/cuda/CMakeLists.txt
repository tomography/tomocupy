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
set_target_properties(cfunc_fourierrec PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_fourierrec.py" DESTINATION tomocupy)
install(TARGETS cfunc_fourierrec LIBRARY DESTINATION tomocupy)

set_source_files_properties(cfunc_lprec.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_lprec LANGUAGE python SOURCES cfunc_lprec.i cfunc_lprec.cu)
swig_link_libraries(cfunc_lprec cudart cufft)
set_target_properties(cfunc_lprec PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_lprec.py" DESTINATION tomocupy)
install(TARGETS cfunc_lprec LIBRARY DESTINATION tomocupy)

set_source_files_properties(cfunc_linerec.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_linerec LANGUAGE python SOURCES cfunc_linerec.i cfunc_linerec.cu)
swig_link_libraries(cfunc_linerec cudart)
set_target_properties(cfunc_linerec PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_linerec.py" DESTINATION tomocupy)
install(TARGETS cfunc_linerec LIBRARY DESTINATION tomocupy)

set_source_files_properties(cfunc_filter.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_filter LANGUAGE python SOURCES cfunc_filter.i cfunc_filter.cu)
swig_link_libraries(cfunc_filter cudart cufft)
set_target_properties(cfunc_filter PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_filter.py" DESTINATION tomocupy)
install(TARGETS cfunc_filter LIBRARY DESTINATION tomocupy)

set_source_files_properties(cfunc_fourierrecfp16.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_fourierrecfp16 LANGUAGE python SOURCES cfunc_fourierrecfp16.i cfunc_fourierrec.cu)
swig_link_libraries(cfunc_fourierrecfp16 cudart cufft)
set_target_properties(cfunc_fourierrecfp16 PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
target_compile_definitions(cfunc_fourierrecfp16 PRIVATE HALF)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_fourierrecfp16.py" DESTINATION tomocupy)
install(TARGETS cfunc_fourierrecfp16 LIBRARY DESTINATION tomocupy)

set_source_files_properties(cfunc_lprecfp16.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_lprecfp16 LANGUAGE python SOURCES cfunc_lprecfp16.i cfunc_lprec.cu)
swig_link_libraries(cfunc_lprecfp16 cudart cufft)
set_target_properties(cfunc_lprecfp16 PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
target_compile_definitions(cfunc_lprecfp16 PRIVATE HALF)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_lprecfp16.py" DESTINATION tomocupy)
install(TARGETS cfunc_lprecfp16 LIBRARY DESTINATION tomocupy)

set_source_files_properties(cfunc_linerecfp16.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_linerecfp16 LANGUAGE python SOURCES cfunc_linerecfp16.i cfunc_linerec.cu)
swig_link_libraries(cfunc_linerecfp16 cudart)
set_target_properties(cfunc_linerecfp16 PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
target_compile_definitions(cfunc_linerecfp16 PRIVATE HALF)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_linerecfp16.py" DESTINATION tomocupy)
install(TARGETS cfunc_linerecfp16 LIBRARY DESTINATION tomocupy)

set_source_files_properties(cfunc_filterfp16.i PROPERTIES CPLUSPLUS ON)
swig_add_library(cfunc_filterfp16 LANGUAGE python SOURCES cfunc_filterfp16.i cfunc_filter.cu)
swig_link_libraries(cfunc_filterfp16 cudart cufft)
set_target_properties(cfunc_filterfp16 PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
target_compile_definitions(cfunc_filterfp16 PRIVATE HALF)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cfunc_filterfp16.py" DESTINATION tomocupy)
install(TARGETS cfunc_filterfp16 LIBRARY DESTINATION tomocupy)

