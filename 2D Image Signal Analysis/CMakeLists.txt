cmake_minimum_required(VERSION 3.6)
project(getImgHistogram)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package( OpenCV REQUIRED )
set(SOURCE_FILES src/main.cpp)
add_executable(main ${SOURCE_FILES})

target_link_libraries( main ${OpenCV_LIBS} )
