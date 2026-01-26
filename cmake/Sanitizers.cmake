# helios/cmake/Sanitizers.cmake
function(helios_enable_sanitizers target ENABLE_ASAN ENABLE_UBSAN)
  if(MSVC)
    # MSVC sanitizers are a different story; ignore for now.
    return()
  endif()

  set(SAN_FLAGS "")
  if(ENABLE_ASAN)
    list(APPEND SAN_FLAGS -fsanitize=address)
  endif()
  if(ENABLE_UBSAN)
    list(APPEND SAN_FLAGS -fsanitize=undefined)
  endif()

  if(SAN_FLAGS)
    target_compile_options(${target} PRIVATE ${SAN_FLAGS} -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE ${SAN_FLAGS})
  endif()
endfunction()