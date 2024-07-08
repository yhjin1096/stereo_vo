cmake_minimum_required(VERSION 3.0.2)
project(StereoVO)

set(OpenCV_DIR /home/cona/open_library/opencv-3.4.16/install/share/OpenCV)

find_package(catkin REQUIRED COMPONENTS
  roscpp
)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES StereoVO
  CATKIN_DEPENDS roscpp
  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)
link_libraries(
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node src/main.cpp)
