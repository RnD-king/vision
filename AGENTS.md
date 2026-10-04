# Vision ROS2 Adapter Agent Guide

## Project role

`vision` is the hardware/runtime adapter for the shared P2P vision system. It
owns ROS2 topics, camera/IMU/depth input, CUDA preprocessing, TensorRT inference,
conversion to shared-core types, visualization, diagnostics, and command
transport. Reusable perception and mission behavior belongs in `vision_core`.

## Architecture boundary

The active pipeline is:

camera / IMU / RGB-D → preprocessing and TensorRT → observation adapters →
`vision_core::MissionController` → direct `ControlCommand` → ROS messages →
P2P executor

Do not reintroduce `MotionCommand`, velocity control, `P2pMotionQuantizer`,
`/cmd_vel`, RL/MuJoCo compatibility, or a duplicated mission FSM. The ROS node
must not independently select mission priority, force transitions, reset
individual controllers, or emulate completion in the real-executor path.

## Configuration ownership

Shared algorithm parameters come only from
`vision_core/config/vision_algorithm.yaml`. ROS/runtime wiring belongs in
`vision/config/vision_params.yaml`, and TensorRT/model selection belongs in
`vision/config/yolo26_runtime.yaml`. Do not duplicate canonical algorithm
defaults in this repository.

## ROS execution contract

`ActionCommand.msg`, `CameraCommand.msg`, and `CommandStatus.msg` are external
interfaces. Preserve action numbers, `action_id`, and ACK/READY/DONE semantics.

For a current action, ACK means the executor accepted and started it. READY does
not cancel or finish it. After READY, vision may publish one queued LINE action
with a new ID. The executor must ACK that ID only after storing the complete
command, then execute it exactly once after current DONE. Repeated messages with
the same ID must be idempotent.

Camera triggers latch while locomotion runs. Publish camera commands only after
locomotion DONE and keep locomotion on HOLD until camera settled feedback.

Internal action feedback is allowed only in an explicit simulation path. Real
robot and tuning paths must wait for the executor's actual ACK/DONE.

## Hardware safety and verification

Do not autonomously launch camera or robot-motion execution without explicit
user authorization. Builds and static checks are safe.

Typical verification:

```bash
source /opt/ros/humble/setup.bash
cd /home/noh/my_cv
CMAKE_PREFIX_PATH=/home/noh/vision_core/install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH} \
  colcon build --packages-select vision --symlink-install --cmake-clean-cache
```

When a change depends on `vision_core`, build, test, and install the core first
so the adapter uses the current headers and library.

## Scope discipline

Inspect the adapter, corresponding core API, and external message consumers;
modify only the owning layer; and avoid unrelated refactors.
