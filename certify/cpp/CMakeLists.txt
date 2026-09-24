# =============================================================================
# Certification drivers (self-certifying GBP, Phase A+): snapshot, freeze, export.
# Built only with -DGSOLVER_BUILD_CERTIFY=ON; they reuse the examples' graph builders.
# =============================================================================
foreach(app certify_charuco certify_stereo certify_chain)
  add_executable(${app} ${app}.cpp)
  target_link_libraries(${app} PRIVATE gsolver::gsolver yaml-cpp)
  target_include_directories(${app} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/examples/cpp)
endforeach()
