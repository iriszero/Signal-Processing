cmake_minimum_required(VERSION 3.6)
project(Signals_Systems_Final)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-O2")

find_package( OpenCV REQUIRED )

set(SOURCE_FILES src/main.cpp)
add_executable(Car_Classifier ${SOURCE_FILES})

target_link_libraries( Car_Classifier ${OpenCV_LIBS} )
