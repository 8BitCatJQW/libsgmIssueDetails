cmake_minimum_required(VERSION 3.1)
project(testSGM)
if (CMAKE_COMPILER_IS_GNUCXX)
	#set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

#set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)


find_package(CUDA REQUIRED)
set(OpenCV_STATIC ON)
set(OpenCV_DIR D:/thirdparty/build/install/win64)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories("$D:/PCL 1.9.1/bin") 


set(openNI D:/openNI)
find_package(openNI REQUIRED)
include_directories(${openNI_INCLUDE_DIRS})





#find_package(Threads)
#find_package(OpenCV REQUIRED core)

add_definitions(-std=c++11)
include_directories(include_sgm)
include_directories(/include)

include_directories($ENV{DaHeng_INCLUDE_DIR})
link_directories($ENV{DaHeng_LIBRARY_DIRS})

FILE(GLOB_RECURSE CPP_SRC_LIST "src/*.cpp")

if (OpenCV_VERSION VERSION_LESS 3.0)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.0")
endif()

configure_file(${CMAKE_SOURCE_DIR}/include_sgm/libsgm_config.h.in
               ${CMAKE_SOURCE_DIR}/include_sgm/libsgm_config.h
)

add_subdirectory(src_sgm)

cuda_add_executable(testSGM main.cpp ${CPP_SRC_LIST})
target_link_libraries(testSGM sgm ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OPENNI_LIBRARY_DIR}  GxIAPI DxImageProc)


