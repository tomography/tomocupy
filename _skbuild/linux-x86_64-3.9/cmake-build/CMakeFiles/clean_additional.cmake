# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Release")
  file(REMOVE_RECURSE
  "src/cuda/CMakeFiles/cfunc_fourierrec.dir/cfunc_fourierrecPYTHON_wrap.cxx"
  "src/cuda/CMakeFiles/cfunc_fourierrecfp16.dir/cfunc_fourierrecfp16PYTHON_wrap.cxx"
  "src/cuda/cfunc_fourierrec.py"
  "src/cuda/cfunc_fourierrecfp16.py"
  )
endif()
