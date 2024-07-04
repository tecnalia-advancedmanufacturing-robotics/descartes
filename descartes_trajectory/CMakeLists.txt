cmake_minimum_required(VERSION 3.5)
project(descartes_trajectory)

# Add support for C++11
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)

set(INCLUDE_DIRS include ${ament_cmake_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRS})

set(LIBRARY_DIRS ${ament_cmake_LIBRARY_DIRS})

link_directories(${LIBRARY_DIRS})

set(LIBS ${ament_cmake_LIBRARIES})

add_compile_options(-std=c++17 -Wall -Wextra)

find_package(ament_cmake
  REQUIRED)

find_package(descartes_core REQUIRED)
find_package(moveit_core REQUIRED)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

# Eigen 3.2 (Wily) only provides EIGEN3_INCLUDE_DIR, not EIGEN3_INCLUDE_DIRS
if(NOT EIGEN3_INCLUDE_DIRS)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()


###########
## Build ##
###########

add_library(${PROJECT_NAME} src/axial_symmetric_pt.cpp
  src/cart_trajectory_pt.cpp src/joint_trajectory_pt.cpp src/trajectory.cpp)

ament_target_dependencies(${PROJECT_NAME} descartes_core moveit_core)

#############
## Install ##
#############

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME})
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

#############
## Testing ##
#############
#TODO: re-enable once I figure out how to test on ROS2
if(BUILD_TESTING)
  # set(UTEST_SRC_FILES test/utest.cpp test/trajectory_pt.cpp)
  # ament_cmake_add_gtest(${PROJECT_NAME}_utest ${UTEST_SRC_FILES})
  # target_compile_definitions(${PROJECT_NAME}_utest
  #   PUBLIC GTEST_USE_OWN_TR1_TUPLE=0)
  # target_link_libraries(${PROJECT_NAME}_utest ${PROJECT_NAME})
endif()

ament_export_dependencies(ament_cmake)
ament_export_include_directories(${INCLUDE_DIRS})
ament_export_libraries(${PROJECT_NAME} ${LIBS})

ament_package()
