cmake_minimum_required(VERSION 3.5)
project(mpc)

# find dependencies
# find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(nav_msgs REQUIRED)

add_executable(path src/path.cpp)
ament_target_dependencies(path rclcpp geometry_msgs visualization_msgs std_msgs builtin_interfaces nav_msgs)

# Install the executable
install(TARGETS
  path
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
 #ament_lint_auto_find_test_dependencies()
endif()

ament_package()
