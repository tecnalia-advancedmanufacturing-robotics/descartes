cmake_minimum_required(VERSION 3.5)
project(descartes_utilities)

# Add support for C++11
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)

set(AMENT_DEPENDENCIES
  descartes_core
  descartes_trajectory
  trajectory_msgs
  eigen_stl_containers
  urdf
  moveit_msgs
  rclcpp
  visualization_msgs
)

foreach(DEP ${AMENT_DEPENDENCIES})
  find_package(${DEP} REQUIRED)
endforeach()


find_package(Eigen3 3.3 REQUIRED)

set(INCLUDE_DIRS include ${ament_cmake_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRS})

set(LIBRARY_DIRS ${ament_cmake_LIBRARY_DIRS})

link_directories(${LIBRARY_DIRS})

set(LIBS ${ament_cmake_LIBRARIES})

add_compile_options(-std=c++17 -Wall -Wextra)

find_package(ament_cmake
  REQUIRED
  COMPONENTS descartes_core descartes_trajectory trajectory_msgs)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS descartes_core descartes_trajectory trajectory_msgs)

add_library(${PROJECT_NAME} src/ros_conversions.cpp)

ament_target_dependencies(${PROJECT_NAME} ${AMENT_DEPENDENCIES})

target_link_libraries(${PROJECT_NAME} ${ament_cmake_LIBRARIES} Eigen3::Eigen)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING
  PATTERN "*.h")

ament_export_dependencies(ament_cmake)
ament_export_include_directories(${INCLUDE_DIRS})
ament_export_libraries(${PROJECT_NAME} ${LIBS})

ament_package()
