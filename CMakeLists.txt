CMAKE_MINIMUM_REQUIRED ( VERSION 3.2.0 )

set(CMAKE_BUILD_TYPE Release)
IF(NOT_ROOT_PROJECT STREQUAL "")
elseif()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/build)
  SET(NOT_ROOT_PROJECT TRUE)
  RETURN()
endif()

PROJECT( slack_prop )
set (VERSION_MAJOR 1)
set (VERSION_MINOR 2)

FILE(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ADD_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${ADD_CMAKE_MODULE_PATH} )
SET (MYLIBS)

SET (SRC_DIR src)
include_directories(${SRC_DIR})

SET (MY_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/")
SET (MY_BIN_PATH "${CMAKE_CURRENT_LIST_DIR}/bin/")
SET (MEX_INSTALL_PATH ${MY_LIB_PATH})

##---------packages--------------------------------
#-------MobileVision-------------------------------

FILE(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ADD_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${ADD_CMAKE_MODULE_PATH})

set(ImageUtilities_DIR $ENV{IMAGEUTILITIES_ROOT})
find_package(ImageUtilities REQUIRED COMPONENTS ndarray)
include_directories(${ImageUtilities_DIR}/src)
include_directories(${IMAGEUTILITIES_INCLUDE_DIR})
SET(MYLIBS ${MYLIBS} ${IMAGEUTILITIES_LIBRARIES})

##---------compiler--------------------------------
include(cmake/compiler_settings.cmake.txt)
clear_flag(CMAKE_CXX_FLAGS "-fpermissive")
add_flags(CUDA_NVCC_FLAGS "-Xptxas=--verbose")
#add_flags(CUDA_NVCC_FLAGS "-res-usage -maxrregcount=128")

#----------print includes--------------------------------
message("INCLUDE:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

##--------build functions--------------------------

include(functions.cmake.txt)

##-----------------------------------------------------------------------------

#SET(SRC
#${SRC_DIR}/slp_driver.cu
#${SRC_DIR}/slp_kernel.cu
#${SRC_DIR}/slp_test.cpp
#)

SET(SRC
${SRC_DIR}/slp_driver.cpp
${SRC_DIR}/slp_kernel.cu
#${SRC_DIR}/ndarray/error.cpp
#${SRC_DIR}/ndarray/ndarray_mem.cpp
#${SRC_DIR}/ndarray/ndarray.cpp
#${SRC_DIR}/ndarray/ndarray_op.cu
)

#message(STATUS "CUDA_NVCC_FLAGS= ${CUDA_NVCC_FLAGS}")

#my_cuda_add_library(slack_prop SHARED ${SRC})
#my_cuda_add_library(slack_prop STATIC ${SRC})

my_cuda_add_library(slack_prop ${SRC})
#my_cuda_add_executable(slp_test ${SRC_DIR}/slp_test.cpp)

#my_cuda_add_executable(corr_FQ_test ${SRC_DIR}/correlation_FQ_test.cpp ${SRC_DIR}/correlation_FQ_kernel.cu)


