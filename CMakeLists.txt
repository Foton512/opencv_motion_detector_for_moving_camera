cmake_minimum_required(VERSION 3.3)
project(motion_tracker_for_moving_camera)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(motion_tracker_for_moving_camera ${SOURCE_FILES})
target_link_libraries(motion_tracker_for_moving_camera ${OpenCV_LIBS})