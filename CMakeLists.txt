cmake_minimum_required (VERSION 2.6)

SET( PROJECT_NAME "OpencvAnisotropy" )
PROJECT( ${PROJECT_NAME} )

SET( TARGET_0     "OpencvAnisotropy"         )
SET( TARGET_0_CPP "test_integral_corr2.cpp"  )

###############################################

IF(NOT CMAKE_BUILD_TYPE)
#    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_BUILD_TYPE "Debug")
ENDIF()

SET(PLATFORM "x32")
IF("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
    set(PLATFORM "x64")
ENDIF()

SET(BUILD_PREFIX "${CMAKE_BUILD_TYPE}_${PLATFORM}")
MESSAGE("build prefix=[${BUILD_PREFIX}]")

###############################################
FIND_PACKAGE( OpenCV REQUIRED )
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

###############################################
set(LIB_UTIL            "util")
add_library( ${LIB_UTIL} STATIC "src/util.cpp" "include/util.h" "include/lut.h")

set(EXT_LIBS    ${LIB_UTIL})

###############################################
ADD_EXECUTABLE("${TARGET_0}_${BUILD_PREFIX}" "${TARGET_0_CPP}")
TARGET_LINK_LIBRARIES("${TARGET_0}_${BUILD_PREFIX}" ${EXT_LIBS} ${OpenCV_LIBS} )

##get_cmake_property(_variableNames VARIABLES)
##foreach (_variableName ${_variableNames})
##    message(STATUS "${_variableName}=${${_variableName}}")
##endforeach()
