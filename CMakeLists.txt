
cmake_minimum_required(VERSION 2.8)
project(test)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(Boost_FOUND)
    message("find Boost")
    include_directories(${Boost_INCLUDE_DIR})
endif()
# add_library(${PROJECT_NAME} test.cpp)
# # add_dependencies(${PROJECT_NAME} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

set(TEST_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libDisplayImage.a)
add_executable(${PROJECT_NAME} test.cpp)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
# target_link_libraries(${PROJECT_NAME} ${TEST_LIBRARIES} ${OpenCV_LIBS} )



install(DIRECTORY include  DESTINATION ./
        # PATTERN ".h" EXCLUDE
        # PATTERN "scripts/*"
        # PATTERN "*.h"
        # PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        #             GROUP_EXECUTE GROUP_READ
                    )



