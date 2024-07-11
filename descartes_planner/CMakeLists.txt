cmake_minimum_required(VERSION 3.5)
project(descartes_planner)

# Add support for C++11
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()

find_package(ament_cmake REQUIRED)

set(INCLUDE_DIRS include ${ament_cmake_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRS})

set(LIBRARY_DIRS ${ament_cmake_LIBRARY_DIRS})

link_directories(${LIBRARY_DIRS})

set(LIBS ${ament_cmake_LIBRARIES})

add_compile_options(-std=c++17)

find_package(ament_cmake
  REQUIRED)

set(AMENT_DEPENDENCIES
  descartes_core
  descartes_trajectory
  pluginlib rclcpp
  eigen_stl_containers
  urdf
  moveit_msgs
  rclcpp
  visualization_msgs
)

find_package(Eigen3 REQUIRED)

foreach(DEP ${AMENT_DEPENDENCIES})
  find_package(${DEP} REQUIRED)
endforeach()

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

# Let's try to use open-mp parallization if we can
find_package(OpenMP)
if(OPENMP_FOUND)
  message(STATUS "Descartes will use OpenMP")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
endif()

# Eigen 3.2 (Wily) only provides EIGEN3_INCLUDE_DIR, not EIGEN3_INCLUDE_DIRS
if(NOT EIGEN3_INCLUDE_DIRS)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

###################################
## ament_cmake specific configuration ##
###################################
set(THIS_PACKAGE_INCLUDE_DEPENDS
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS descartes_core descartes_trajectory rclcpp
  DEPENDS Boost EIGEN3)

###########
## Build ##
###########

## DescartesTrajectoryPt lib
add_library(${PROJECT_NAME} src/dense_planner.cpp
  src/ladder_graph_dag_search.cpp src/planning_graph.cpp src/plugin_init.cpp
  src/sparse_planner.cpp)

ament_target_dependencies(${PROJECT_NAME} ${AMENT_DEPENDENCIES})
target_link_libraries(${PROJECT_NAME} ${ament_cmake_LIBRARIES} ${OpenMP_FLAGS} Eigen3::Eigen)

target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_FLAGS})

# add_dependencies(${PROJECT_NAME} ${ament_cmake_EXPORTED_TARGETS})

#############
## Install ##
#############
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(FILES ${PROJECT_NAME}_plugins.xml
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME})

ament_export_dependencies(ament_cmake)
ament_export_include_directories(${INCLUDE_DIRS})
ament_export_libraries(${PROJECT_NAME} ${LIBS})

ament_package()
