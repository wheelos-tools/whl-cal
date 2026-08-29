cmake_minimum_required(VERSION 3.16)

project(gril_native VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# This is intentionally a source-build-only research target.  Do not add
# install rules until the documented validation and license gates are approved.
install(CODE [[
  message(FATAL_ERROR
    "GRIL native distribution is blocked: the full A/B rotation gate, "
    "independent physical acceptance, and formal license/provenance review "
    "must be approved before an installable package is created.")
]])

set(
  GRIL_PINNED_EIGEN_INCLUDE_DIR "" CACHE PATH
  "Eigen 3.3.7 include directory for pinned GRIL frontend equivalence"
)
if(GRIL_PINNED_EIGEN_INCLUDE_DIR)
  set(
    GRIL_EIGEN_VERSION_HEADER
    "${GRIL_PINNED_EIGEN_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h"
  )
  if(NOT EXISTS "${GRIL_EIGEN_VERSION_HEADER}")
    message(FATAL_ERROR
      "GRIL_PINNED_EIGEN_INCLUDE_DIR does not contain Eigen headers")
  endif()
  file(READ "${GRIL_EIGEN_VERSION_HEADER}" GRIL_EIGEN_VERSION_CONTENT)
  foreach(EIGEN_COMPONENT WORLD MAJOR MINOR)
    if(EIGEN_COMPONENT STREQUAL "WORLD")
      set(EXPECTED_VERSION 3)
    elseif(EIGEN_COMPONENT STREQUAL "MAJOR")
      set(EXPECTED_VERSION 3)
    else()
      set(EXPECTED_VERSION 7)
    endif()
    if(NOT GRIL_EIGEN_VERSION_CONTENT MATCHES
       "#define EIGEN_${EIGEN_COMPONENT}_VERSION[ \t]+${EXPECTED_VERSION}")
      message(FATAL_ERROR
        "GRIL frontend equivalence requires Eigen 3.3.7 exactly")
    endif()
  endforeach()
  add_library(gril_pinned_eigen INTERFACE)
  include_directories(BEFORE SYSTEM "${GRIL_PINNED_EIGEN_INCLUDE_DIR}")
  target_include_directories(
    gril_pinned_eigen SYSTEM BEFORE
    INTERFACE "${GRIL_PINNED_EIGEN_INCLUDE_DIR}"
  )
  set(GRIL_EIGEN_TARGET gril_pinned_eigen)
else()
  find_package(Eigen3 3.3.7 EXACT REQUIRED NO_MODULE)
  set(GRIL_EIGEN_TARGET Eigen3::Eigen)
endif()
find_package(Ceres 2.0 QUIET)
find_package(PCL 1.8 REQUIRED COMPONENTS common filters)
find_package(Threads REQUIRED)

foreach(PCL_TARGET IN LISTS PCL_LIBRARIES)
  if(TARGET ${PCL_TARGET})
    set_property(TARGET ${PCL_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS "")
  endif()
endforeach()
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=32)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Log")

add_library(gril_native_fusion
  include/Fusion/FusionAhrs.c
  include/Fusion/FusionCompass.c
  include/Fusion/FusionOffset.c
)
add_library(GRIL::fusion ALIAS gril_native_fusion)
target_include_directories(gril_native_fusion
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(Ceres_FOUND)
  add_library(gril_native
    include/Gril_Calib/BatchTrace.cpp
    include/Gril_Calib/Gril_Calib.cpp
  )
  add_library(GRIL::native ALIAS gril_native)
  target_include_directories(gril_native
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_compile_definitions(
    gril_native PRIVATE ROOT_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\"
  )
  target_link_libraries(
    gril_native PUBLIC ${GRIL_EIGEN_TARGET} Ceres::ceres
  )
  target_compile_features(gril_native PUBLIC cxx_std_14)
else()
  message(STATUS
    "Ceres Solver not found; omitting the calibration-core targets. "
    "Patchwork++ and frontend trace targets remain available.")
endif()

add_library(gril_native_velodyne
  include/Gril_Calib/VelodynePreprocess.cpp
)
add_library(GRIL::velodyne ALIAS gril_native_velodyne)
target_include_directories(gril_native_velodyne
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  gril_native_velodyne PUBLIC ${GRIL_EIGEN_TARGET} ${PCL_LIBRARIES}
)
target_compile_options(
  gril_native_velodyne PUBLIC ${PCL_DEFINITIONS}
)
target_compile_features(gril_native_velodyne PUBLIC cxx_std_14)

add_library(gril_native_frontend
  include/Gril_Calib/FrontendCore.cpp
)
add_library(GRIL::frontend ALIAS gril_native_frontend)
target_include_directories(gril_native_frontend
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  gril_native_frontend PUBLIC ${GRIL_EIGEN_TARGET} ${PCL_LIBRARIES}
)
target_compile_options(
  gril_native_frontend PUBLIC ${PCL_DEFINITIONS}
)
target_compile_features(gril_native_frontend PUBLIC cxx_std_14)

add_library(gril_native_odometry
  include/Gril_Calib/LidarOdometry.cpp
  include/ikd-Tree/ikd_Tree.cpp
)
add_library(GRIL::odometry ALIAS gril_native_odometry)
target_include_directories(gril_native_odometry
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  gril_native_odometry
  PUBLIC
    GRIL::frontend
    ${GRIL_EIGEN_TARGET}
    ${PCL_LIBRARIES}
    Threads::Threads
)
target_compile_options(
  gril_native_odometry PUBLIC ${PCL_DEFINITIONS}
)
target_compile_features(gril_native_odometry PUBLIC cxx_std_14)

add_library(gril_native_patchwork INTERFACE)
add_library(GRIL::patchwork ALIAS gril_native_patchwork)
target_include_directories(gril_native_patchwork
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  gril_native_patchwork
  INTERFACE ${GRIL_EIGEN_TARGET} ${PCL_LIBRARIES}
)
target_compile_options(
  gril_native_patchwork INTERFACE ${PCL_DEFINITIONS}
)

if(Ceres_FOUND)
  add_executable(gril_native_smoke smoke/gril_native_smoke.cpp)
  target_link_libraries(gril_native_smoke PRIVATE GRIL::native)

  add_executable(gril_native_batch src/gril_batch.cpp)
  target_link_libraries(gril_native_batch PRIVATE GRIL::native)
endif()

add_executable(gril_native_preprocess_trace src/gril_preprocess_trace.cpp)
target_link_libraries(
  gril_native_preprocess_trace PRIVATE GRIL::velodyne
)

add_executable(gril_native_frontend_cv_trace src/gril_frontend_cv_trace.cpp)
target_link_libraries(
  gril_native_frontend_cv_trace PRIVATE GRIL::frontend
)

add_executable(
  gril_native_frontend_event_trace src/gril_frontend_event_trace.cpp
)
target_link_libraries(
  gril_native_frontend_event_trace
  PRIVATE GRIL::frontend GRIL::velodyne
)

add_executable(gril_native_ground_trace src/gril_ground_trace.cpp)
target_link_libraries(
  gril_native_ground_trace PRIVATE GRIL::patchwork
)

add_executable(gril_native_odometry_trace src/gril_odometry_trace.cpp)
target_link_libraries(
  gril_native_odometry_trace PRIVATE GRIL::odometry
)

if(Ceres_FOUND)
  add_library(gril_native_full_frontend
    include/Gril_Calib/FullFrontend.cpp
  )
  add_library(GRIL::full_frontend ALIAS gril_native_full_frontend)
  target_include_directories(gril_native_full_frontend
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${PCL_INCLUDE_DIRS}
  )
  target_link_libraries(
    gril_native_full_frontend
    PUBLIC
      GRIL::native
      GRIL::velodyne
      GRIL::odometry
      GRIL::patchwork
      GRIL::fusion
      ${PCL_LIBRARIES}
  )
  target_compile_options(
    gril_native_full_frontend PUBLIC ${PCL_DEFINITIONS}
  )
  target_compile_features(
    gril_native_full_frontend PUBLIC cxx_std_14
  )

  add_executable(
    gril_native_full_frontend_exec src/gril_full_frontend.cpp
  )
  set_target_properties(
    gril_native_full_frontend_exec
    PROPERTIES OUTPUT_NAME gril_native_full_frontend
  )
  target_link_libraries(
    gril_native_full_frontend_exec PRIVATE GRIL::full_frontend
  )
endif()

include(CTest)
if(BUILD_TESTING)
  if(Ceres_FOUND)
    add_test(NAME gril_native_smoke COMMAND gril_native_smoke)
    add_executable(gril_native_trace_test tests/batch_trace_test.cpp)
    target_link_libraries(gril_native_trace_test PRIVATE GRIL::native)
    add_test(
      NAME gril_native_trace_round_trip
      COMMAND gril_native_trace_test round_trip
    )
    add_test(
      NAME gril_native_trace_rejection
      COMMAND gril_native_trace_test rejection
    )
    add_executable(
      gril_native_full_test tests/full_frontend_test.cpp
    )
    target_link_libraries(
      gril_native_full_test PRIVATE GRIL::full_frontend
    )
    foreach(mode hard_time rollback source_scan_pairing)
      add_test(
        NAME gril_native_full_${mode}
        COMMAND gril_native_full_test ${mode}
      )
    endforeach()
  endif()
  add_executable(
    gril_native_velodyne_test tests/velodyne_preprocess_test.cpp
  )
  target_link_libraries(
    gril_native_velodyne_test PRIVATE GRIL::velodyne
  )
  foreach(mode
      supplied_zero_first
      fallback_azimuth
      nonfinite
      ring_stride_blind
      cut_frame)
    add_test(
      NAME gril_native_velodyne_${mode}
      COMMAND gril_native_velodyne_test ${mode}
    )
  endforeach()
  add_executable(
    gril_native_frontend_test tests/frontend_core_test.cpp
  )
  target_link_libraries(
    gril_native_frontend_test PRIVATE GRIL::frontend
  )
  foreach(mode synchronization propagation)
    add_test(
      NAME gril_native_frontend_${mode}
      COMMAND gril_native_frontend_test ${mode}
    )
  endforeach()
  add_executable(
    gril_native_odometry_test tests/lidar_odometry_test.cpp
  )
  target_link_libraries(
    gril_native_odometry_test PRIVATE GRIL::odometry
  )
  foreach(mode
      pinned_config
      map_initialization
      ekf_update
      determinism
      pcl_1_10_voxel_order)
    add_test(
      NAME gril_native_odometry_${mode}
      COMMAND gril_native_odometry_test ${mode}
    )
  endforeach()
  add_executable(
    gril_native_patchwork_test tests/patchworkpp_test.cpp
  )
  target_link_libraries(
    gril_native_patchwork_test PRIVATE GRIL::patchwork
  )
  add_test(
    NAME gril_native_patchwork_smoke
    COMMAND gril_native_patchwork_test
  )
endif()
