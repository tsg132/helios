# helios/cmake/LTO.cmake
include(CheckIPOSupported)

function(helios_enable_lto target ENABLE_LTO)
  if(NOT ENABLE_LTO)
    return()
  endif()

  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if(ipo_supported)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "IPO/LTO not supported: ${ipo_error}")
  endif()
endfunction()