cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)
SET(CMAKE_BUILD_TYPE Debug)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Check C++11 or C++0x support
add_definitions(-DCOMPILEDWITHC11)
if( CMAKE_BUILD_TYPE MATCHES "Debug" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -g -march=native -Wno-deprecated-declarations")
else( CMAKE_BUILD_TYPE MATCHES "Release" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -O3 -march=native -Wno-deprecated-declarations")
endif( CMAKE_BUILD_TYPE MATCHES "Debug" )

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        src/Converter.cpp
        src/Frame.cpp
        src/FrameDrawer.cpp
        src/KeyFrame.cpp
        src/KeyFrameDatabase.cpp
        src/LocalMapping.cpp
        src/LoopClosing.cpp
        src/Map.cpp
        src/MapDrawer.cpp
        src/MapPoint.cpp
        src/Optimizer.cpp
        src/ORBextractor.cpp
        src/ORBmatcher.cpp
        src/PnPsolver.cpp
        src/Sim3Solver.cpp
        src/System.cpp
        src/Tracking.cpp
        src/Viewer.cpp
        src/DUOReader.cpp
)

set(DUO_SDK ${PROJECT_SOURCE_DIR}/Thirdparty/SDK)
include_directories(${DUO_SDK}/include)
if (CMAKE_SYSTEM_PROCESSOR STREQUAL armv7l OR CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
    set(DUOLib_ARCH arm)
else ()
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(DUOLib_ARCH x64)
    else ()
        set(DUOLib_ARCH x86)
    endif ()
endif ()
find_library(DUO_LIBRARY DUO HINTS ${DUO_SDK}/linux/${DUOLib_ARCH})

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${DUO_LIBRARY}
)

# Build tools
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Stereo)
add_executable(bin_vocabulary Stereo/bin_vocabulary.cpp)
target_link_libraries(bin_vocabulary ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so ${OpenCV_LIBS})

add_executable(DUOStereo Stereo/DUOStereo.cpp)
target_link_libraries(DUOStereo ${PROJECT_NAME})