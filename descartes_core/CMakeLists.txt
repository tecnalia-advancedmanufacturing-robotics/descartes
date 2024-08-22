cmake_minimum_required(VERSION 3.5)
project(descartes_core)

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

find_package(ament_cmake REQUIRED COMPONENTS cmake_modules rclcpp)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

# Eigen 3.2 (Wily) only provides EIGEN3_INCLUDE_DIR, not EIGEN3_INCLUDE_DIRS
if(NOT EIGEN3_INCLUDE_DIRS)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

find_package(rclcpp REQUIRED)
find_package(Boost REQUIRED)
# find_package(EIGEN3 REQUIRED)
# foreach(dep_pkg ${THIS_PACKAGE_INCLUDE_DEPENDS})
#     find_package(${dep_pkg} REQUIRED)
# endforeach()

###########
## Build ##
###########

add_library(${PROJECT_NAME} SHARED src/trajectory_id.cpp)

target_link_libraries(${PROJECT_NAME} ${ament_cmake_LIBRARIES}
  ${EIGEN3_LIBRARIES})

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

ament_export_dependencies(ament_cmake)
ament_export_include_directories(${INCLUDE_DIRS})
ament_export_libraries(${PROJECT_NAME} ${LIBS})

ament_package()
