cmake_minimum_required(VERSION 3.0.2)
project(test)





## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Debug)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
# find_package(catkin REQUIRED COMPONENTS
#     roscpp
#     rospy
#     std_msgs
#     message_generation
#     twc_tuya_msgs
#     geometry_msgs
#     nav_msgs
#     genmsg
#     actionlib_msgs
#     actionlib
# )

# find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# pkg_check_modules(LZ4 REQUIRED liblz4)
include_directories(${LZ4_INCLUDE_DIRS})
link_directories(${LZ4_LIBRARY_DIRS})




###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
#     include
#     include/tuya
#     include/tuya/hal
#     include/tuya/hal/driver
#     include/tuya/hal/system
    ${LZ4_INCLUDE_DIRS}
#     ${OpenCV_INCLUDE_DIRS}

)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/twc_tuya_iot.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_node
    test.cpp

)



## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
#   ${TUYA_LIBRARIES}
#   ${LIBRT}
#   ${LIBM}
  ${LZ4_LIBRARIES}
  ${OpenCV_LIBRARIES}

)


