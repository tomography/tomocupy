include_directories(include)

if(${EXTENSION_WRAPPER} STREQUAL swig)
  message("-- Using SWIG as the Python extension wrapper.")
  add_subdirectory(cuda)
else()
  message(
    FATAL_ERROR
    "${EXTENSION_WRAPPER} is not a valid Python extension wrapper."
  )
endif()