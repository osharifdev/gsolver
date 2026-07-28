# =============================================================================
# Reproduction tooling for "Breaking Time" (arXiv:2606.06250v1)
#
# Standalone project: compiles the UNMODIFIED gsolver library sources together
# with new instrumented evaluation apps. The shipped library and example apps
# are not touched; solver schedule / precision / factor definitions unchanged.
# Rerun (visualisation) is deliberately NOT linked — no visualisation anywhere.
#
# Configure:
#   cmake -S reproduction -B reproduction/build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=$HOME/gsolver-deps/install \
#         -DEIGEN_SRC=$PWD/build/_deps/eigen-src
# =============================================================================
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(gsolver_reproduction CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(EIGEN_SRC "${REPO_ROOT}/build/_deps/eigen-src" CACHE PATH "Eigen source dir (same 3.4.0 used by main build)")

find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)

# Same gsolver source list as gsolver/CMakeLists.txt (identical flags: -O3 -DNDEBUG)
set(GSOLVER_SOURCES
  ${REPO_ROOT}/gsolver/src/core/gbp_solver.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/variables/se3_point.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/variables/se3_pose.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/variables/se3_posevel.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/factors/se3_point_equality.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/factors/se3_posevel_gp.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/factors/se3_posevel_point.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/factors/se3_posevel_stereo_point.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/factors/se3_posevel_pose.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/factors/se3_posevel_posevel.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/factors/se3_posevel_prior.cpp
  ${REPO_ROOT}/gsolver/src/graph/types/instantiations.cpp
)
add_library(gsolver_repro STATIC ${GSOLVER_SOURCES})
target_include_directories(gsolver_repro PUBLIC
  ${REPO_ROOT}/gsolver/include
  ${EIGEN_SRC}
)
target_include_directories(gsolver_repro PRIVATE ${REPO_ROOT}/gsolver/src)
target_link_libraries(gsolver_repro PUBLIC OpenMP::OpenMP_CXX)

function(add_repro_app name)
  add_executable(${name} src/${name}.cpp)
  target_include_directories(${name} PRIVATE ${REPO_ROOT}/examples/cpp)
  target_link_libraries(${name} PRIVATE gsolver_repro yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX)
endfunction()

add_repro_app(repro_synthetic)
add_repro_app(repro_charuco)
add_repro_app(repro_stereo)
add_repro_app(repro_bench)
add_repro_app(torus_generator)
