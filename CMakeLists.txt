cmake_minimum_required(VERSION 3.18)
project(vision)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_VISION_DEFAULT_CUDA_ARCH 87)
  else()
    set(_VISION_DEFAULT_CUDA_ARCH 86)
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${_VISION_DEFAULT_CUDA_ARCH} CACHE STRING "CUDA architecture: 86 for RTX Ampere PC, 87 for Jetson Orin")
endif()

if(EXISTS "/usr/local/cuda-12.4")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.4" CACHE PATH "CUDA toolkit root")
  set(CUDA_BIN_PATH         "/usr/local/cuda-12.4" CACHE PATH "CUDA bin path")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ActionCommand.msg"
  "msg/CameraCommand.msg"
  "msg/CommandStatus.msg"
)

message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "OpenCV_VERSION=${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

find_package(CUDAToolkit REQUIRED)

# /home/noh/vision_core에 별도로 설치한 ROS 비의존 공통 계산 라이브러리.
set(_SHARED_VISION_CORE_DEFAULT "/home/noh/vision_core/install")
if(DEFINED ENV{VISION_CORE_ROOT} AND NOT "$ENV{VISION_CORE_ROOT}" STREQUAL "")
  set(_SHARED_VISION_CORE_DEFAULT "$ENV{VISION_CORE_ROOT}")
endif()
set(SHARED_VISION_CORE_ROOT "${_SHARED_VISION_CORE_DEFAULT}" CACHE PATH
    "shared_vision_core install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${SHARED_VISION_CORE_ROOT}")
find_package(shared_vision_core CONFIG REQUIRED)

# ---------- TensorRT ----------
set(TENSORRT_ROOT "" CACHE PATH "TensorRT root directory (optional)")

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${TENSORRT_ROOT} /usr /usr/local /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu
  PATH_SUFFIXES include include/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER
  NAMES nvinfer
  HINTS ${TENSORRT_ROOT} /usr /usr/local /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

find_library(TENSORRT_NVINFER_PLUGIN
  NAMES nvinfer_plugin
  HINTS ${TENSORRT_ROOT} /usr /usr/local /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find TensorRT headers (NvInfer.h).")
endif()
if(NOT TENSORRT_NVINFER)
  message(FATAL_ERROR "Could not find TensorRT library: nvinfer.")
endif()
if(NOT TENSORRT_NVINFER_PLUGIN)
  message(FATAL_ERROR "Could not find TensorRT library: nvinfer_plugin.")
endif()

add_library(vision_core
  src/yolo_trt_engine.cpp
  src/line_detection_adapter.cpp
  src/yolo_preprocess.cu
)

set_target_properties(vision_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(vision_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(vision_core
  shared_vision_core::shared_vision_core
  ${OpenCV_LIBS}
  ${TENSORRT_NVINFER}
  ${TENSORRT_NVINFER_PLUGIN}
  CUDA::cudart
)

# source 없이 설치된 ROS2 실행 파일만 띄워도 독립 코어를 찾을 수 있게 한다.
set_target_properties(vision_core PROPERTIES
  BUILD_RPATH "${SHARED_VISION_CORE_ROOT}/lib"
  INSTALL_RPATH "${SHARED_VISION_CORE_ROOT}/lib"
)

ament_target_dependencies(vision_core
  rclcpp sensor_msgs geometry_msgs cv_bridge
)

add_executable(line_perception_node
  src/line_perception_node.cpp
)

target_link_libraries(line_perception_node
  vision_core
)
rosidl_get_typesupport_target(vision_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(line_perception_node "${vision_typesupport_target}")

set_target_properties(line_perception_node PROPERTIES
  BUILD_RPATH "${SHARED_VISION_CORE_ROOT}/lib"
  # libvision_core.so is installed beside this executable in lib/vision.
  # $ORIGIN lets the dynamic loader find it without a manual LD_LIBRARY_PATH.
  INSTALL_RPATH "$ORIGIN;${SHARED_VISION_CORE_ROOT}/lib"
)

ament_target_dependencies(line_perception_node
  rclcpp ament_index_cpp sensor_msgs geometry_msgs cv_bridge
)

install(TARGETS
  vision_core
  line_perception_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
