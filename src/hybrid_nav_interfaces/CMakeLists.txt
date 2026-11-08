cmake_minimum_required(VERSION 3.8)
project(hybrid_nav_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Frontier.msg"
  "msg/FrontierArray.msg"
  "srv/EvaluateFrontiers.srv"
  "srv/VerifyTarget.srv"
  "srv/MatchCandidate3DGS.srv"
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
)

ament_package()
