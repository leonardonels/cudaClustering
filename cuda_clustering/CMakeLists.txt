cmake_minimum_required(VERSION 3.8)
project(clustering)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(CUDA  REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(PCL REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Find include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(/opt/ros/humble/include/pcl_ros)
include_directories(/opt/ros/humble/include/pcl_conversions)
include_directories(/opt/ros/humble/include/message_filters)
include_directories(/opt/ros/humble/include/pcl_msgs)   
include_directories(/opt/ros/humble/include/tf2)   
include_directories(/opt/ros/humble/include/tf2_ros)   
include_directories(/opt/ros/humble/include/tf2_msgs) 

include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_definitions(-std=c++17 -g -lboost_thread-mt)

add_library(${PROJECT_NAME} src/controller_node.cpp)

add_executable(clustering_node
  src/controller_node.cpp
  src/main.cpp
  src/clustering/cuda_clustering.cpp
  src/filtering/cuda_filtering.cpp
  src/segmentation/cuda_segmentation.cpp
)
target_link_libraries(clustering_node
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  cuda_clustering
  cuda_filtering
  cuda_segmentation
  pointcloud_converter
)
target_include_directories(clustering_node PUBLIC include)

add_library(libcudacluster SHARED IMPORTED)
set_target_properties( libcudacluster PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libcudacluster.so )

add_library(libcudafilter SHARED IMPORTED)
set_target_properties( libcudafilter PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libcudafilter.so )

add_library(libcudasegmentation SHARED IMPORTED)
set_target_properties( libcudasegmentation PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libcudasegmentation.so )

link_directories(libcudacluster)
link_directories(libcudafilter)
link_directories(libcudasegmentation)

add_library(cuda_clustering src/clustering/cuda_clustering.cpp)
add_library(cluster_filtering src/clustering/cluster_filtering/dimension_filter.cpp)
add_library(cuda_filtering src/filtering/cuda_filtering.cpp)
add_library(cuda_segmentation src/segmentation/cuda_segmentation.cpp)

# Optimized pointcloud converter library with -O3
add_library(pointcloud_converter src/utils/pointcloud_converter.cpp)
target_compile_options(pointcloud_converter PRIVATE -O3 -march=native -mtune=native -ffast-math)
target_include_directories(pointcloud_converter PUBLIC include)
ament_target_dependencies(pointcloud_converter
  sensor_msgs
  pcl_conversions
  PCL
)

ament_target_dependencies(cluster_filtering
  geometry_msgs
)

target_link_libraries(cuda_clustering
  libcudacluster
  cluster_filtering
)
ament_target_dependencies(cuda_clustering
  rclcpp
  visualization_msgs
)
target_link_libraries(cuda_filtering
  libcudafilter
)
ament_target_dependencies(cuda_filtering
  rclcpp
)
target_link_libraries(cuda_segmentation
  libcudasegmentation
)
ament_target_dependencies(cuda_segmentation
  rclcpp
)

target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  cuda_clustering
  cuda_filtering
  cuda_segmentation
  pointcloud_converter
)
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  PCL
)

target_link_libraries(${PROJECT_NAME}_node 
  ${PROJECT_NAME}
)

target_include_directories(clustering PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(clustering PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS clustering_node
  DESTINATION lib/${PROJECT_NAME})


# Install launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  lib
  DESTINATION share/${PROJECT_NAME}
)

# Install config directory
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
