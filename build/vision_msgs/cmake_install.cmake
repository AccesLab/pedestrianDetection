# Install script for directory: /home/mashuk/catkin_ws/src/vision_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/mashuk/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vision_msgs/msg" TYPE FILE FILES
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/BoundingBox2D.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/BoundingBox3D.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/Classification2D.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/Classification3D.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/Detection2DArray.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/Detection2D.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/Detection3DArray.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/Detection3D.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/ObjectHypothesis.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/ObjectHypothesisWithPose.msg"
    "/home/mashuk/catkin_ws/src/vision_msgs/msg/VisionInfo.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vision_msgs/cmake" TYPE FILE FILES "/home/mashuk/catkin_ws/build/vision_msgs/catkin_generated/installspace/vision_msgs-msg-paths.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/mashuk/catkin_ws/devel/include/vision_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/mashuk/catkin_ws/devel/share/roseus/ros/vision_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/mashuk/catkin_ws/devel/share/common-lisp/ros/vision_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/mashuk/catkin_ws/devel/share/gennodejs/ros/vision_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  execute_process(COMMAND "/usr/bin/python" -m compileall "/home/mashuk/catkin_ws/devel/lib/python2.7/dist-packages/vision_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/mashuk/catkin_ws/devel/lib/python2.7/dist-packages/vision_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/mashuk/catkin_ws/build/vision_msgs/catkin_generated/installspace/vision_msgs.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vision_msgs/cmake" TYPE FILE FILES "/home/mashuk/catkin_ws/build/vision_msgs/catkin_generated/installspace/vision_msgs-msg-extras.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vision_msgs/cmake" TYPE FILE FILES
    "/home/mashuk/catkin_ws/build/vision_msgs/catkin_generated/installspace/vision_msgsConfig.cmake"
    "/home/mashuk/catkin_ws/build/vision_msgs/catkin_generated/installspace/vision_msgsConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vision_msgs" TYPE FILE FILES "/home/mashuk/catkin_ws/src/vision_msgs/package.xml")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/vision_msgs" TYPE DIRECTORY FILES "/home/mashuk/catkin_ws/src/vision_msgs/include/vision_msgs/" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/mashuk/catkin_ws/build/vision_msgs/test/cmake_install.cmake")

endif()

