cmake_minimum_required(VERSION 3.20)
project(yolo_srv)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yolo_interfaces REQUIRED)

find_library(ONNXRUNTIME_LIB onnxruntime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(yolo_srv_node src/yolo_srv.cpp)
target_include_directories(yolo_srv_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE})

target_link_libraries(yolo_srv_node PUBLIC
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${yolo_interfaces_TARGETS}
)

target_link_libraries(yolo_srv_node PRIVATE
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB}
)

install(TARGETS yolo_srv_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
