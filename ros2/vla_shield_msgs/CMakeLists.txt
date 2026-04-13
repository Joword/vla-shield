cmake_minimum_required(VERSION 3.8)
project(vla_shield_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ActionProposal.msg"
  "msg/SafetyDecision.msg"
  "msg/RiskTelemetry.msg"
  "srv/ExplainBlock.srv"
  DEPENDENCIES builtin_interfaces
)

ament_package()
