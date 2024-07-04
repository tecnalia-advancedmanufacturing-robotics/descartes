cmake_minimum_required(VERSION 3.5)
project(descartes_moveit)

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

find_package(ament_cmake REQUIRED)
set(AMENT_DEPENDENCIES
  descartes_core descartes_trajectory moveit_core
  moveit_ros_planning pluginlib
)
foreach(DEP ${AMENT_DEPENDENCIES})
  find_package(${DEP} REQUIRED)
endforeach()
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

# Eigen 3.2 (Wily) only provides EIGEN3_INCLUDE_DIR, not EIGEN3_INCLUDE_DIRS
if(NOT EIGEN3_INCLUDE_DIRS)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()


###########
## Build ##
###########

## Descartes MoveIt lib
add_library(${PROJECT_NAME} src/ikfast_moveit_state_adapter.cpp
  src/moveit_state_adapter.cpp src/plugin_init.cpp src/seed_search.cpp)
ament_target_dependencies(${PROJECT_NAME} ${AMENT_DEPENDENCIES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)


#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib/${PROJECT_NAME})

install(FILES moveit_adapter_plugins.xml
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME})

ament_export_dependencies(ament_cmake)
ament_export_include_directories(${INCLUDE_DIRS})
ament_export_libraries(${PROJECT_NAME} ${LIBS})

ament_package()
