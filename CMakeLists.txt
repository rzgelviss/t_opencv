
cmake_minimum_required(VERSION 2.8)
project(test)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

message(${OpenCV_INCLUDE_DIRS})

# include_directories(/home/twc/cproject/opencv/opencv/opencv_contrib/modules/xfeatures2d/include)
option(HAVE_OPENCV_XFEATURES2D "HAVE_OPENCV_XFEATURES2D" ON)
add_compile_options(-DHAVE_OPENCV_XFEATURES2D)



if(Boost_FOUND)
    message("find Boost")
    include_directories(${Boost_INCLUDE_DIR})
endif()
# add_library(${PROJECT_NAME} test.cpp)
# # add_dependencies(${PROJECT_NAME} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(TEST_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libDisplayImage.a)
# set(XFEATURES2D_LIBRARIES /usr/local/lib/libopencv_xfeatures2d.so)
add_executable(${PROJECT_NAME} test_meanshift.cpp)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(${PROJECT_NAME} ${TEST_LIBRARIES} ${OpenCV_LIBS} )



install(DIRECTORY include  DESTINATION ./
        # PATTERN ".h" EXCLUDE
        # PATTERN "scripts/*"
        # PATTERN "*.h"
        # PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        #             GROUP_EXECUTE GROUP_READ
                    )



