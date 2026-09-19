# Vision ROS2 Adapter Agent Guide

## Project role

`vision` is the hardware/runtime adapter layer for the shared vision system.

Its responsibilities include:

- ROS2 subscriptions and publications
- camera and IMU input
- CUDA image preprocessing
- TensorRT YOLO inference
- conversion from detector outputs to shared-core types
- ROS command transport
- runtime/device-specific configuration
- visualization and diagnostics

Reusable perception, mission, and control algorithms belong in `vision_core`.

## Architecture boundary

Do not duplicate `vision_core` algorithms in this repository.

In particular, avoid implementing mission logic, target-selection policy, line-control equations, mission state transitions, or action-lifecycle policy directly in `line_perception_node.cpp`.

The intended pipeline is:

camera / IMU
→ preprocessing
→ TensorRT YOLO
→ detection adapters
→ `vision_core::MissionController`
→ `ControlCommand`
→ ROS2 messages

`line_perception_node` is primarily an orchestration and transport layer.

## vision_core dependency

The active implementation depends on the separately installed `shared_vision_core`.

Do not copy shared-core source code into this repository as a workaround.

When behavior belongs in both the real robot and simulator, implement it in `vision_core` first and keep only the adapter here.

## Active source vs legacy

`legacy/` is reference-only.

It is not part of the active build.

Do not modify or revive legacy code unless the user explicitly asks for it.

Do not use legacy code as the authoritative implementation when an equivalent implementation exists in `vision_core`.

## Configuration ownership

Shared algorithm parameters belong in:

`vision_core/config/vision_algorithm.yaml`

ROS/runtime-specific parameters belong in:

`vision/config/vision_params.yaml`

YOLO/TensorRT runtime/model-selection parameters belong in:

`vision/config/yolo26_runtime.yaml`

Do not duplicate the same algorithm constant across multiple layers without a clear compatibility reason.

## ROS interface compatibility

The following messages form an external execution contract:

- `ActionCommand.msg`
- `CameraCommand.msg`
- `CommandStatus.msg`

Do not change message fields, action IDs, ACK/DONE semantics, or command lifecycle behavior casually.

Before changing an interface, inspect all producers and consumers.

Mission actions and locomotion/P2P actions must retain their established action-ID lifecycle semantics.

## Mission ownership

`MissionController` in `vision_core` is the single mission-level decision entry point.

The ROS node should not independently:

- select mission priority
- reset individual mission controllers
- force mission transitions
- emulate controller completion state

It should provide observations/feedback and transport the returned command.

## TensorRT / CUDA

TensorRT engine files and generated model artifacts are not source files and should not be committed.

Keep PC and Jetson differences in runtime/build configuration rather than duplicating algorithm logic.

Do not assume a CUDA architecture or TensorRT installation outside the existing build/config contract without checking the target machine.

## Hardware execution

Do not autonomously launch the camera, robot command topics, or physical-motion execution unless explicitly requested.

Builds and static checks are safe; robot-moving runtime commands are not.

## Build verification

Typical workspace build:

`source /opt/ros/humble/setup.bash`

`cd /home/noh/my_cv`

`colcon build --packages-select vision --symlink-install`

Before claiming a modification is valid, at minimum ensure the affected target builds.

If a change depends on `vision_core`, verify that the correct installed `vision_core` version is being used.

## Scope discipline

For targeted fixes:

1. inspect the relevant node/adapter/config,
2. inspect the corresponding `vision_core` API,
3. determine which repository owns the behavior,
4. modify only the owning layer,
5. verify the build.

Avoid unrelated cleanup or refactoring unless requested.
