cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(sparsecoding)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "C++11 support is required, but not supported by ${CMAKE_CXX_COMPILER}.")
endif()

find_package(OpenCV REQUIRED)

file(COPY images DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(INCLUDE include)
include_directories(${INCLUDE})

set(HEADERS include/App.hpp include/DictionaryLearner.hpp)
set(SOURCES src/Main.cpp src/App.cpp src/DictionaryLearner.cpp)
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
