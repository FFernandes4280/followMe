cmake_minimum_required(VERSION 3.10)
project(followme_yolo)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(followMe followMe.cpp)
target_link_libraries(followMe ${OpenCV_LIBS})
