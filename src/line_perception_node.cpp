// vision/src/line_perception_node.cpp
//
// 1. line_perception_node.cpp
//    전체 ROS 노드 시작점. 이미지/IMU를 받고 전체 파이프라인을 호출함.

// 2. yolo_trt_engine.cpp
//    line_perception_node가 YOLO 추론을 요청하면 실행됨.
//    엔진 로드, 입력/출력 메모리 관리, TensorRT enqueue, 후처리 담당.

// 3. yolo_preprocess.cu
//    yolo_trt_engine.cpp 내부 Preprocess()에서 호출됨.
//    YOLO 입력용으로 BGR -> RGB, HWC -> CHW, float [0,1], letterbox padding
//    수행.

// 4. yolo_trt_engine.cpp
//    TensorRT 추론 후 output [x1,y1,x2,y2,conf,class]를 원본 이미지 좌표 bbox로
//    복원.

// 5. shared_vision_core/MissionController::StepPerception
//    detection 전처리, 선택적 IMU 좌표 보정, line 특징·진입 판정·활성
//    미션 잠금·내부 단계·최종 ControlCommand를 계산한다.

// 7. line_perception_node.cpp
//    core가 반환한 최종 속도/액션/카메라 명령을 ROS 토픽으로 전달한다.

// 터미널1: ros2 run my_cv val_image_publisher
// 터미널2: ros2 run vision line_perception_node

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/parameter_map.hpp>
#include <rclcpp/rclcpp.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <sys/utsname.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "vision_core/config_loader.hpp"
#include "vision_core/mission_controller.hpp"

#include "vision/yolo_trt_engine.hpp"
#include "vision/msg/action_command.hpp"
#include "vision/msg/camera_command.hpp"
#include "vision/msg/command_status.hpp"

namespace vision {

namespace {

bool IsJetsonTarget() {
  struct utsname info {};
  if (uname(&info) != 0) {
    return false;
  }
  const std::string machine(info.machine);
  return machine == "aarch64" || machine == "arm64";
}

std::string RuntimeConfigPath() {
  if (const char *env_path = std::getenv("YOLO26_RUNTIME_CONFIG"); env_path != nullptr && env_path[0] != '\0') {
    return std::string(env_path);
  }
  return ament_index_cpp::get_package_share_directory("vision") + "/config/yolo26_runtime.yaml";
}

std::string ReadRuntimeString(const std::string &param_name) {
  try {
    const auto map = rclcpp::parameter_map_from_yaml_file(RuntimeConfigPath());
    for (const auto &node_name : {"/yolo26_runtime", "yolo26_runtime"}) {
      const auto it = map.find(node_name);
      if (it == map.end()) {
        continue;
      }
      for (const auto &param : it->second) {
        if (param.get_name() == param_name) {
          return param.as_string();
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[WARN] Failed to load YOLO26 runtime config: " << e.what() << std::endl;
  }
  return {};
}

std::string DefaultYolo26EnginePath() {
  if (const char *env_path = std::getenv("YOLO26_ENGINE_PATH"); env_path != nullptr && env_path[0] != '\0') {
    return std::string(env_path);
  }

  return ReadRuntimeString(IsJetsonTarget() ? "jetson_engine_path" : "pc_engine_path");
}

vision_core::MissionControllerConfig LoadSharedAlgorithmDefaults() {
  const std::string path = vision_core::DefaultAlgorithmConfigPath();
  auto config = vision_core::LoadAlgorithmConfig(path);
  std::cerr << "[INFO] Loaded shared vision algorithm config: " << path
            << std::endl;
  return config;
}

} // namespace

class LinePerceptionNode final : public rclcpp::Node {
public:
  LinePerceptionNode() : rclcpp::Node("line_perception_node") {
    const vision_core::MissionControllerConfig algorithm_defaults =
        LoadSharedAlgorithmDefaults();
    declare_parameter<std::string>("image_topic", "/camera/color/image_raw"); // 입력 이미지 토픽
    declare_parameter<std::string>("depth_topic", "/camera/aligned_depth_to_color/image_raw");
    declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
    declare_parameter<std::string>("imu_topic", "/camera/imu_tilt");          // IMU roll/pitch 토픽
    declare_parameter<std::string>("prev_cmd_topic", "/jandi_vision/cmd_vel");   // 이전 속도 참조 토픽
    declare_parameter<std::string>("cmd_topic", "/jandi_vision/cmd_vel");        // 최종 속도 publish 토픽
    declare_parameter<std::string>("action_cmd_topic", "/jandi_vision/action_cmd");
    declare_parameter<std::string>("action_status_topic", "/jandi_vision/action_status");
    declare_parameter<std::string>("camera_cmd_topic", "/jandi_vision/camera_cmd");
    declare_parameter<std::string>("camera_status_topic", "/jandi_vision/camera_status");
    declare_parameter<std::string>("engine_path", DefaultYolo26EnginePath()); // TensorRT engine 경로
    declare_parameter<int>("line_class_id", algorithm_defaults.line_detection.class_id); // line YOLO class id
    declare_parameter<int>("ball_class_id", algorithm_defaults.object_targets.ball_class_id); // ball YOLO class id
    declare_parameter<int>("goal_class_id", algorithm_defaults.object_targets.goal_class_id); // goal YOLO class id
    declare_parameter<int>("backboard_class_id", algorithm_defaults.object_targets.backboard_class_id); // backboard YOLO class id
    declare_parameter<int>("hurdle_class_id", algorithm_defaults.object_targets.hurdle_class_id); // hurdle YOLO class id
    declare_parameter<double>("conf_thres", algorithm_defaults.line_detection.confidence); // YOLO 1차 confidence threshold
    declare_parameter<double>("ball_conf_thres", algorithm_defaults.object_targets.ball_confidence); // ball target 2차 confidence threshold
    declare_parameter<double>("goal_conf_thres", algorithm_defaults.object_targets.goal_confidence); // goal target 2차 confidence threshold
    declare_parameter<double>("backboard_conf_thres", algorithm_defaults.object_targets.backboard_confidence); // backboard target 2차 confidence threshold
    declare_parameter<double>("hurdle_conf_thres", algorithm_defaults.object_targets.hurdle_confidence); // hurdle target 2차 confidence threshold
    declare_parameter<int>("object_min_box_width", static_cast<int>(algorithm_defaults.object_targets.min_box_width));
    declare_parameter<int>("object_min_box_height", static_cast<int>(algorithm_defaults.object_targets.min_box_height));
    declare_parameter<double>("fx", 600.0);                      // camera intrinsic fx
    declare_parameter<double>("fy", 600.0);                      // camera intrinsic fy
    declare_parameter<double>("cx", 320.0);                      // camera intrinsic cx
    declare_parameter<double>("cy", 240.0);                      // camera intrinsic cy
    declare_parameter<bool>("use_imu_rectification", true);      // IMU 기반 픽셀 보정 사용 여부
    declare_parameter<bool>("assume_zero_imu", false);           // roll/pitch를 0으로 가정
    declare_parameter<double>("imu_abs_limit_deg", 45.0);        // IMU roll/pitch clamp 각도
    // core가 확정한 객체만 표시하는 기본 디버그 화면/상태 패널이다.
    declare_parameter<bool>("show_debug_view", true);
    // YOLO의 필터 전 raw detection bbox를 별도 창에서 확인할 때만 켠다.
    declare_parameter<bool>("show_yolo_debug_view", false);
    declare_parameter<double>("inference_hz", 1.0 / std::max(algorithm_defaults.line_observation_dt, 1e-6));
    declare_parameter<int>("max_centers", algorithm_defaults.line_features.max_centers);
    declare_parameter<double>("lookahead_delta_v_px", algorithm_defaults.line_features.lookahead_delta_v_px);
    declare_parameter<double>("lookahead_alpha_normal", algorithm_defaults.line_features.lookahead_alpha_normal);
    declare_parameter<double>("lookahead_alpha_recovery", algorithm_defaults.line_features.lookahead_alpha_recovery);
    declare_parameter<double>("recover_enter_nvis", algorithm_defaults.line_features.recover_enter_nvis);
    declare_parameter<double>("recover_exit_nvis", algorithm_defaults.line_features.recover_exit_nvis);
    declare_parameter<double>("recover_enter_u", algorithm_defaults.line_features.recover_enter_u);
    declare_parameter<double>("recover_exit_u", algorithm_defaults.line_features.recover_exit_u);
    declare_parameter<double>("vx_prev_min", 0.0);               // 이전 vx clamp 최소값
    declare_parameter<double>("vx_prev_max", 1.2);               // 이전 vx clamp 최대값
    declare_parameter<double>("wz_prev_min", -1.9);              // 이전 wz clamp 최소값
    declare_parameter<double>("wz_prev_max", 1.9);               // 이전 wz clamp 최대값
    declare_parameter<bool>("enable_rule_controller", true);     // 최종 cmd publish 여부
    // 실제 카메라 request/feedback ROS I/O는 아직 없으므로 안전하게 기본 OFF.
    // true이면 코어의 시간 추정 compatibility 경로로만 임시 시퀀스를 확인한다.
    declare_parameter<bool>("enable_ball_controller", algorithm_defaults.enable_ball);
    declare_parameter<bool>("enable_hurdle_controller", algorithm_defaults.enable_hurdle);
    declare_parameter<bool>("enable_goal_controller", algorithm_defaults.enable_goal);
    // all은 위 enable_* 값을 따르고, 나머지는 지정한 알고리즘 하나만 실행한다.
    declare_parameter<std::string>("algorithm_mode", "all");
    // 실제 로봇은 모든 보행을 /action_cmd로 출력한다. velocity는 호환 시험용이다.
    declare_parameter<std::string>(
        "locomotion_backend",
        algorithm_defaults.command.locomotion_backend ==
                vision_core::LocomotionBackend::kP2pAction
            ? "p2p"
            : "velocity");
    declare_parameter<double>("p2p_forward_deadband", algorithm_defaults.command.p2p.forward_deadband);
    declare_parameter<double>("p2p_lateral_deadband", algorithm_defaults.command.p2p.lateral_deadband);
    declare_parameter<double>("p2p_yaw_deadband", algorithm_defaults.command.p2p.yaw_deadband);
    declare_parameter<double>("p2p_long_forward_vx", algorithm_defaults.command.p2p.long_forward_vx);
    declare_parameter<double>("p2p_curve_yaw_threshold", algorithm_defaults.command.p2p.curve_yaw_threshold);
    declare_parameter<double>("p2p_turn_in_place_vx_max", algorithm_defaults.command.p2p.turn_in_place_vx_max);
    declare_parameter<double>("p2p_lateral_dominance_ratio", algorithm_defaults.command.p2p.lateral_dominance_ratio);
    declare_parameter<double>("p2p_fine_forward_deadband", algorithm_defaults.command.p2p_fine.forward_deadband);
    declare_parameter<double>("p2p_fine_lateral_deadband", algorithm_defaults.command.p2p_fine.lateral_deadband);
    declare_parameter<double>("p2p_fine_yaw_deadband", algorithm_defaults.command.p2p_fine.yaw_deadband);
    declare_parameter<double>("p2p_fine_long_forward_vx", algorithm_defaults.command.p2p_fine.long_forward_vx);
    declare_parameter<double>("p2p_fine_curve_yaw_threshold", algorithm_defaults.command.p2p_fine.curve_yaw_threshold);
    declare_parameter<double>("p2p_fine_turn_in_place_vx_max", algorithm_defaults.command.p2p_fine.turn_in_place_vx_max);
    declare_parameter<double>("p2p_fine_lateral_dominance_ratio", algorithm_defaults.command.p2p_fine.lateral_dominance_ratio);
    declare_parameter<double>("p2p_recovery_forward_deadband", algorithm_defaults.command.p2p_recovery.forward_deadband);
    declare_parameter<double>("p2p_recovery_lateral_deadband", algorithm_defaults.command.p2p_recovery.lateral_deadband);
    declare_parameter<double>("p2p_recovery_yaw_deadband", algorithm_defaults.command.p2p_recovery.yaw_deadband);
    declare_parameter<double>("p2p_recovery_long_forward_vx", algorithm_defaults.command.p2p_recovery.long_forward_vx);
    declare_parameter<double>("p2p_recovery_curve_yaw_threshold", algorithm_defaults.command.p2p_recovery.curve_yaw_threshold);
    declare_parameter<double>("p2p_recovery_turn_in_place_vx_max", algorithm_defaults.command.p2p_recovery.turn_in_place_vx_max);
    declare_parameter<double>("p2p_recovery_lateral_dominance_ratio", algorithm_defaults.command.p2p_recovery.lateral_dominance_ratio);
    // 실제 카메라 모터 ROS I/O 연결 전 단독 알고리즘 시험용 endpoint 피드백이다.
    declare_parameter<bool>("simulate_camera_feedback", false);
    // 카메라 기반 판단만 시험할 때 Action 실행기의 ACK/DONE을 내부 모사한다.
    declare_parameter<bool>("simulate_decision", false);
    declare_parameter<double>("simulate_action_duration_sec", 0.50);
    declare_parameter<bool>("enable_command_transport", true);
    declare_parameter<double>("rl_stop_duration_sec", algorithm_defaults.ball.rl_stop_duration_sec);
    declare_parameter<double>("max_depth_age_sec", 0.20);
    declare_parameter<double>("backboard_min_depth_m", algorithm_defaults.backboard_min_depth_m);
    declare_parameter<double>("backboard_max_depth_m", algorithm_defaults.backboard_max_depth_m);
    declare_parameter<double>("goal_fine_adjust_start_z_m", algorithm_defaults.goal.fine_adjust_start_z_m);
    declare_parameter<double>("goal_hoop_radius_m", algorithm_defaults.goal.hoop_radius_m);
    declare_parameter<double>("goal_throwing_range_m", algorithm_defaults.goal.throwing_range_m);
    declare_parameter<double>("goal_position_tolerance_m", algorithm_defaults.goal.position_tolerance_m);
    declare_parameter<double>("cmd_vx_min", algorithm_defaults.line.cmd_vx_min);
    declare_parameter<double>("cmd_vx_max", algorithm_defaults.line.cmd_vx_max);
    declare_parameter<double>("cmd_wz_min", algorithm_defaults.line.cmd_wz_min);
    declare_parameter<double>("cmd_wz_max", algorithm_defaults.line.cmd_wz_max);
    declare_parameter<double>("rule_v_base", algorithm_defaults.line.v_base);
    declare_parameter<double>("rule_k_u", algorithm_defaults.line.k_u);
    declare_parameter<double>("rule_k_slope", algorithm_defaults.line.k_slope);
    declare_parameter<double>("rule_k_v_u", algorithm_defaults.line.k_v_u);
    declare_parameter<double>("rule_k_v_slope", algorithm_defaults.line.k_v_slope);
    declare_parameter<double>("rule_dv_max", algorithm_defaults.line.dv_max);
    declare_parameter<double>("rule_dw_max", algorithm_defaults.line.dw_max);
    declare_parameter<double>("rule_recover_vx", algorithm_defaults.line.recover_vx);
    declare_parameter<double>("rule_recover_wz", algorithm_defaults.line.recover_wz);
    declare_parameter<double>("rule_low_visible_n", algorithm_defaults.line.low_visible_n);
    declare_parameter<double>("rule_no_visible_n", algorithm_defaults.line.no_visible_n);
    declare_parameter<double>("rule_low_visible_vx", algorithm_defaults.line.low_visible_vx);
    declare_parameter<double>("rule_no_visible_vx", algorithm_defaults.line.no_visible_vx);
    declare_parameter<double>("rule_low_visible_wz_decay", algorithm_defaults.line.low_visible_wz_decay);
    declare_parameter<double>("rule_no_visible_wz_decay", algorithm_defaults.line.no_visible_wz_decay);
    declare_parameter<int>("ball_stable_window", algorithm_defaults.ball.stable_window);
    declare_parameter<int>("ball_stable_min_hits", algorithm_defaults.ball.stable_min_hits);
    declare_parameter<int>("ball_lost_frames", algorithm_defaults.ball.lost_frames);
    declare_parameter<double>("ball_smooth_alpha", algorithm_defaults.ball.smooth_alpha);
    declare_parameter<double>("ball_far_u_des_norm", algorithm_defaults.ball.far_u_des_norm);
    declare_parameter<double>("ball_far_vx", algorithm_defaults.ball.far_vx);
    declare_parameter<double>("ball_far_vx_min", algorithm_defaults.ball.far_vx_min);
    declare_parameter<double>("ball_far_wz_max", algorithm_defaults.ball.far_wz_max);
    declare_parameter<double>("ball_far_heading_gain", algorithm_defaults.ball.far_heading_gain);
    declare_parameter<double>("ball_far_slow_by_turn", algorithm_defaults.ball.far_slow_by_turn);
    declare_parameter<double>("ball_far_dv_max", algorithm_defaults.ball.far_dv_max);
    declare_parameter<double>("ball_far_dw_max", algorithm_defaults.ball.far_dw_max);
    declare_parameter<double>("ball_far_speed_scale", algorithm_defaults.ball.far_speed_scale);
    declare_parameter<double>("ball_tilt_down_v_norm", algorithm_defaults.ball.tilt_down_v_norm);
    declare_parameter<int>("ball_tilt_down_window", algorithm_defaults.ball.tilt_down_window);
    declare_parameter<int>("ball_tilt_down_min_hits", algorithm_defaults.ball.tilt_down_min_hits);
    declare_parameter<double>("ball_tilt_down_h_norm", algorithm_defaults.ball.tilt_down_h_norm);
    declare_parameter<double>("camera_tilt_duration_sec", algorithm_defaults.ball.camera_tilt_duration_sec);
    declare_parameter<double>("camera_settle_sec", algorithm_defaults.ball.camera_settle_sec);
    declare_parameter<double>("camera_return_duration_sec", algorithm_defaults.ball.camera_return_duration_sec);
    declare_parameter<double>("camera_motion_timeout_sec", algorithm_defaults.ball.camera_motion_timeout_sec);
    declare_parameter<int>("ball_hold_cmd_window", algorithm_defaults.ball.hold_cmd_window);
    declare_parameter<double>("ball_hold_vx_min", algorithm_defaults.ball.hold_vx_min);
    declare_parameter<double>("ball_hold_vx_max", algorithm_defaults.ball.hold_vx_max);
    declare_parameter<double>("ball_hold_wz_max", algorithm_defaults.ball.hold_wz_max);
    declare_parameter<double>("ball_hold_default_vx", algorithm_defaults.ball.hold_default_vx);
    declare_parameter<double>("ball_tilt_walk_speed_scale", algorithm_defaults.ball.tilt_walk_speed_scale);
    declare_parameter<double>("ball_tilt_walk_vx_max", algorithm_defaults.ball.tilt_walk_vx_max);
    declare_parameter<double>("ball_fine_adjust_placeholder_vx", algorithm_defaults.ball.fine_adjust_placeholder_vx);
    declare_parameter<double>("ball_fine_adjust_placeholder_sec", algorithm_defaults.ball.fine_adjust_placeholder_duration_sec);
    declare_parameter<double>("ball_pickup_placeholder_sec", algorithm_defaults.ball.pickup_placeholder_duration_sec);
    declare_parameter<double>("ball_ignore_duration_sec", algorithm_defaults.ball.ball_ignore_duration_sec);
    declare_parameter<double>("ball_near_target_u_norm", algorithm_defaults.ball.near_target_u_norm);
    declare_parameter<double>("ball_near_target_v_norm", algorithm_defaults.ball.near_target_v_norm);
    declare_parameter<double>("ball_near_kx", algorithm_defaults.ball.near_kx);
    declare_parameter<double>("ball_near_ky", algorithm_defaults.ball.near_ky);
    declare_parameter<double>("ball_near_wz_gain", algorithm_defaults.ball.near_wz_gain);
    declare_parameter<double>("ball_near_vx_max", algorithm_defaults.ball.near_vx_max);
    declare_parameter<double>("ball_near_vy_max", algorithm_defaults.ball.near_vy_max);
    declare_parameter<double>("ball_near_wz_max", algorithm_defaults.ball.near_wz_max);
    declare_parameter<double>("ball_near_x_tol", algorithm_defaults.ball.near_x_tol);
    declare_parameter<double>("ball_near_y_tol", algorithm_defaults.ball.near_y_tol);
    declare_parameter<bool>("ball_near_use_lateral", algorithm_defaults.ball.near_use_lateral);

    image_topic_ = get_parameter("image_topic").as_string();
    depth_topic_ = get_parameter("depth_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();
    imu_topic_ = get_parameter("imu_topic").as_string();
    prev_cmd_topic_ = get_parameter("prev_cmd_topic").as_string();
    cmd_topic_ = get_parameter("cmd_topic").as_string();
    action_cmd_topic_ = get_parameter("action_cmd_topic").as_string();
    action_status_topic_ = get_parameter("action_status_topic").as_string();
    camera_cmd_topic_ = get_parameter("camera_cmd_topic").as_string();
    camera_status_topic_ = get_parameter("camera_status_topic").as_string();
    engine_path_ = get_parameter("engine_path").as_string();
    line_class_id_ = static_cast<int>(get_parameter("line_class_id").as_int());
    ball_class_id_ = static_cast<int>(get_parameter("ball_class_id").as_int());
    goal_class_id_ = static_cast<int>(get_parameter("goal_class_id").as_int());
    backboard_class_id_ = static_cast<int>(get_parameter("backboard_class_id").as_int());
    hurdle_class_id_ = static_cast<int>(get_parameter("hurdle_class_id").as_int());
    conf_thres_ = static_cast<float>(get_parameter("conf_thres").as_double());
    ball_conf_thres_ = static_cast<float>(get_parameter("ball_conf_thres").as_double());
    goal_conf_thres_ = static_cast<float>(get_parameter("goal_conf_thres").as_double());
    backboard_conf_thres_ = static_cast<float>(get_parameter("backboard_conf_thres").as_double());
    hurdle_conf_thres_ = static_cast<float>(get_parameter("hurdle_conf_thres").as_double());
    use_imu_rectification_ = get_parameter("use_imu_rectification").as_bool();
    assume_zero_imu_ = get_parameter("assume_zero_imu").as_bool();
    imu_abs_limit_rad_ = Deg2Rad(get_parameter("imu_abs_limit_deg").as_double());
    show_debug_view_ = get_parameter("show_debug_view").as_bool();
    show_yolo_debug_view_ =
        get_parameter("show_yolo_debug_view").as_bool();
    inference_hz_ = get_parameter("inference_hz").as_double();
    vx_prev_min_ = get_parameter("vx_prev_min").as_double();
    vx_prev_max_ = get_parameter("vx_prev_max").as_double();
    wz_prev_min_ = get_parameter("wz_prev_min").as_double();
    wz_prev_max_ = get_parameter("wz_prev_max").as_double();
    enable_rule_controller_ = get_parameter("enable_rule_controller").as_bool();
    enable_ball_controller_ = get_parameter("enable_ball_controller").as_bool();
    enable_hurdle_controller_ = get_parameter("enable_hurdle_controller").as_bool();
    enable_goal_controller_ = get_parameter("enable_goal_controller").as_bool();
    algorithm_mode_ = get_parameter("algorithm_mode").as_string();
    locomotion_backend_ = get_parameter("locomotion_backend").as_string();
    simulate_camera_feedback_ = get_parameter("simulate_camera_feedback").as_bool();
    simulate_decision_ = get_parameter("simulate_decision").as_bool();
    simulate_action_duration_sec_ =
        get_parameter("simulate_action_duration_sec").as_double();
    enable_command_transport_ = get_parameter("enable_command_transport").as_bool();
    max_depth_age_sec_ = get_parameter("max_depth_age_sec").as_double();
    if (algorithm_mode_ != "all" && algorithm_mode_ != "line" && algorithm_mode_ != "ball" &&
        algorithm_mode_ != "hurdle" && algorithm_mode_ != "goal") {
      throw std::invalid_argument("algorithm_mode must be one of: all, line, ball, hurdle, goal");
    }
    if (locomotion_backend_ != "velocity" && locomotion_backend_ != "p2p") {
      throw std::invalid_argument(
          "locomotion_backend must be one of: velocity, p2p");
    }
    vision_core::RuleConfig line_cmd_cfg = algorithm_defaults.line;
    line_cmd_cfg.cmd_vx_min = get_parameter("cmd_vx_min").as_double();
    line_cmd_cfg.cmd_vx_max = get_parameter("cmd_vx_max").as_double();
    line_cmd_cfg.cmd_wz_min = get_parameter("cmd_wz_min").as_double();
    line_cmd_cfg.cmd_wz_max = get_parameter("cmd_wz_max").as_double();
    line_cmd_cfg.v_base = get_parameter("rule_v_base").as_double();
    line_cmd_cfg.k_u = get_parameter("rule_k_u").as_double();
    line_cmd_cfg.k_slope = get_parameter("rule_k_slope").as_double();
    line_cmd_cfg.k_v_u = get_parameter("rule_k_v_u").as_double();
    line_cmd_cfg.k_v_slope = get_parameter("rule_k_v_slope").as_double();
    line_cmd_cfg.dv_max = get_parameter("rule_dv_max").as_double();
    line_cmd_cfg.dw_max = get_parameter("rule_dw_max").as_double();
    line_cmd_cfg.recover_vx = get_parameter("rule_recover_vx").as_double();
    line_cmd_cfg.recover_wz = get_parameter("rule_recover_wz").as_double();
    line_cmd_cfg.low_visible_n = get_parameter("rule_low_visible_n").as_double();
    line_cmd_cfg.no_visible_n = get_parameter("rule_no_visible_n").as_double();
    line_cmd_cfg.low_visible_vx = get_parameter("rule_low_visible_vx").as_double();
    line_cmd_cfg.no_visible_vx = get_parameter("rule_no_visible_vx").as_double();
    line_cmd_cfg.low_visible_wz_decay = get_parameter("rule_low_visible_wz_decay").as_double();
    line_cmd_cfg.no_visible_wz_decay = get_parameter("rule_no_visible_wz_decay").as_double();

    vision_core::BallConfig ball_cmd_cfg = algorithm_defaults.ball;
    ball_cmd_cfg.stable_window = static_cast<int>(get_parameter("ball_stable_window").as_int());
    ball_cmd_cfg.stable_min_hits = static_cast<int>(get_parameter("ball_stable_min_hits").as_int());
    ball_cmd_cfg.lost_frames = static_cast<int>(get_parameter("ball_lost_frames").as_int());
    ball_cmd_cfg.smooth_alpha = get_parameter("ball_smooth_alpha").as_double();
    ball_cmd_cfg.far_u_des_norm = get_parameter("ball_far_u_des_norm").as_double();
    ball_cmd_cfg.far_vx = get_parameter("ball_far_vx").as_double();
    ball_cmd_cfg.far_vx_min = get_parameter("ball_far_vx_min").as_double();
    ball_cmd_cfg.far_wz_max = get_parameter("ball_far_wz_max").as_double();
    ball_cmd_cfg.far_heading_gain = get_parameter("ball_far_heading_gain").as_double();
    ball_cmd_cfg.far_slow_by_turn = get_parameter("ball_far_slow_by_turn").as_double();
    ball_cmd_cfg.far_dv_max = get_parameter("ball_far_dv_max").as_double();
    ball_cmd_cfg.far_dw_max = get_parameter("ball_far_dw_max").as_double();
    ball_cmd_cfg.far_speed_scale = get_parameter("ball_far_speed_scale").as_double();
    ball_cmd_cfg.tilt_down_v_norm = get_parameter("ball_tilt_down_v_norm").as_double();
    ball_cmd_cfg.tilt_down_window = static_cast<int>(get_parameter("ball_tilt_down_window").as_int());
    ball_cmd_cfg.tilt_down_min_hits = static_cast<int>(get_parameter("ball_tilt_down_min_hits").as_int());
    ball_cmd_cfg.tilt_down_h_norm = get_parameter("ball_tilt_down_h_norm").as_double();
    ball_cmd_cfg.camera_tilt_duration_sec = get_parameter("camera_tilt_duration_sec").as_double();
    ball_cmd_cfg.camera_settle_sec = get_parameter("camera_settle_sec").as_double();
    ball_cmd_cfg.camera_return_duration_sec = get_parameter("camera_return_duration_sec").as_double();
    ball_cmd_cfg.camera_motion_timeout_sec = get_parameter("camera_motion_timeout_sec").as_double();
    ball_cmd_cfg.hold_cmd_window = static_cast<int>(get_parameter("ball_hold_cmd_window").as_int());
    ball_cmd_cfg.hold_vx_min = get_parameter("ball_hold_vx_min").as_double();
    ball_cmd_cfg.hold_vx_max = get_parameter("ball_hold_vx_max").as_double();
    ball_cmd_cfg.hold_wz_max = get_parameter("ball_hold_wz_max").as_double();
    ball_cmd_cfg.hold_default_vx = get_parameter("ball_hold_default_vx").as_double();
    ball_cmd_cfg.tilt_walk_speed_scale = get_parameter("ball_tilt_walk_speed_scale").as_double();
    ball_cmd_cfg.tilt_walk_vx_max = get_parameter("ball_tilt_walk_vx_max").as_double();
    ball_cmd_cfg.fine_adjust_placeholder_vx = get_parameter("ball_fine_adjust_placeholder_vx").as_double();
    ball_cmd_cfg.fine_adjust_placeholder_duration_sec = get_parameter("ball_fine_adjust_placeholder_sec").as_double();
    ball_cmd_cfg.pickup_placeholder_duration_sec = get_parameter("ball_pickup_placeholder_sec").as_double();
    ball_cmd_cfg.rl_stop_duration_sec = get_parameter("rl_stop_duration_sec").as_double();
    ball_cmd_cfg.ball_ignore_duration_sec = get_parameter("ball_ignore_duration_sec").as_double();
    ball_cmd_cfg.near_target_u_norm = get_parameter("ball_near_target_u_norm").as_double();
    ball_cmd_cfg.near_target_v_norm = get_parameter("ball_near_target_v_norm").as_double();
    ball_cmd_cfg.near_kx = get_parameter("ball_near_kx").as_double();
    ball_cmd_cfg.near_ky = get_parameter("ball_near_ky").as_double();
    ball_cmd_cfg.near_wz_gain = get_parameter("ball_near_wz_gain").as_double();
    ball_cmd_cfg.near_vx_max = get_parameter("ball_near_vx_max").as_double();
    ball_cmd_cfg.near_vy_max = get_parameter("ball_near_vy_max").as_double();
    ball_cmd_cfg.near_wz_max = get_parameter("ball_near_wz_max").as_double();
    ball_cmd_cfg.near_x_tol = get_parameter("ball_near_x_tol").as_double();
    ball_cmd_cfg.near_y_tol = get_parameter("ball_near_y_tol").as_double();
    ball_cmd_cfg.near_use_lateral = get_parameter("ball_near_use_lateral").as_bool();

    const double fx = get_parameter("fx").as_double();
    const double fy = get_parameter("fy").as_double();
    const double cx = get_parameter("cx").as_double();
    const double cy = get_parameter("cy").as_double();

    vision_core::FeatureConfig line_feature_cfg = algorithm_defaults.line_features;
    line_feature_cfg.max_centers = static_cast<int>(get_parameter("max_centers").as_int());
    line_feature_cfg.image_center_u = cx;
    line_feature_cfg.lookahead_delta_v_px = get_parameter("lookahead_delta_v_px").as_double();
    line_feature_cfg.lookahead_alpha_normal = get_parameter("lookahead_alpha_normal").as_double();
    line_feature_cfg.lookahead_alpha_recovery = get_parameter("lookahead_alpha_recovery").as_double();
    line_feature_cfg.recover_enter_nvis = get_parameter("recover_enter_nvis").as_double();
    line_feature_cfg.recover_exit_nvis = get_parameter("recover_exit_nvis").as_double();
    line_feature_cfg.recover_enter_u = get_parameter("recover_enter_u").as_double();
    line_feature_cfg.recover_exit_u = get_parameter("recover_exit_u").as_double();

    if (engine_path_.empty()) {
      throw std::runtime_error("YOLO TensorRT engine 경로(engine_path)가 비어 있습니다.");
    }

    yolo_ = std::make_unique<YoloTrtEngine>(engine_path_, 640, 640, conf_thres_);
    vision_core::HurdleConfig hurdle_cmd_cfg = algorithm_defaults.hurdle;
    hurdle_cmd_cfg.rl_stop_duration_sec = get_parameter("rl_stop_duration_sec").as_double();
    hurdle_cmd_cfg.camera_motion_timeout_sec =
        get_parameter("camera_motion_timeout_sec").as_double();
    vision_core::GoalConfig goal_cmd_cfg = algorithm_defaults.goal;
    goal_cmd_cfg.rl_stop_duration_sec = get_parameter("rl_stop_duration_sec").as_double();
    goal_cmd_cfg.camera_motion_timeout_sec =
        get_parameter("camera_motion_timeout_sec").as_double();
    goal_cmd_cfg.fine_adjust_start_z_m =
        get_parameter("goal_fine_adjust_start_z_m").as_double();
    goal_cmd_cfg.hoop_radius_m =
        get_parameter("goal_hoop_radius_m").as_double();
    goal_cmd_cfg.throwing_range_m =
        get_parameter("goal_throwing_range_m").as_double();
    goal_cmd_cfg.position_tolerance_m =
        get_parameter("goal_position_tolerance_m").as_double();
    vision_core::MissionControllerConfig mission_config = algorithm_defaults;
    mission_config.line_features = line_feature_cfg;
    mission_config.line = line_cmd_cfg;
    mission_config.ball = ball_cmd_cfg;
    mission_config.hurdle = hurdle_cmd_cfg;
    mission_config.goal = goal_cmd_cfg;
    mission_config.line_detection.class_id = line_class_id_;
    mission_config.line_detection.confidence = conf_thres_;
    mission_config.object_targets.ball_class_id = ball_class_id_;
    mission_config.object_targets.goal_class_id = goal_class_id_;
    mission_config.object_targets.backboard_class_id = backboard_class_id_;
    mission_config.object_targets.hurdle_class_id = hurdle_class_id_;
    mission_config.object_targets.ball_confidence = ball_conf_thres_;
    mission_config.object_targets.goal_confidence = goal_conf_thres_;
    mission_config.object_targets.backboard_confidence =
        backboard_conf_thres_;
    mission_config.object_targets.hurdle_confidence = hurdle_conf_thres_;
    mission_config.object_targets.min_box_width =
        get_parameter("object_min_box_width").as_int();
    mission_config.object_targets.min_box_height =
        get_parameter("object_min_box_height").as_int();
    mission_config.backboard_min_depth_m =
        get_parameter("backboard_min_depth_m").as_double();
    mission_config.backboard_max_depth_m =
        get_parameter("backboard_max_depth_m").as_double();
    mission_config.command.locomotion_backend =
        locomotion_backend_ == "p2p"
            ? vision_core::LocomotionBackend::kP2pAction
            : vision_core::LocomotionBackend::kVelocity;
    mission_config.command.p2p.forward_deadband =
        get_parameter("p2p_forward_deadband").as_double();
    mission_config.command.p2p.lateral_deadband =
        get_parameter("p2p_lateral_deadband").as_double();
    mission_config.command.p2p.yaw_deadband =
        get_parameter("p2p_yaw_deadband").as_double();
    mission_config.command.p2p.long_forward_vx =
        get_parameter("p2p_long_forward_vx").as_double();
    mission_config.command.p2p.curve_yaw_threshold =
        get_parameter("p2p_curve_yaw_threshold").as_double();
    mission_config.command.p2p.turn_in_place_vx_max =
        get_parameter("p2p_turn_in_place_vx_max").as_double();
    mission_config.command.p2p.lateral_dominance_ratio =
        get_parameter("p2p_lateral_dominance_ratio").as_double();
    mission_config.command.p2p_fine.forward_deadband =
        get_parameter("p2p_fine_forward_deadband").as_double();
    mission_config.command.p2p_fine.lateral_deadband =
        get_parameter("p2p_fine_lateral_deadband").as_double();
    mission_config.command.p2p_fine.yaw_deadband =
        get_parameter("p2p_fine_yaw_deadband").as_double();
    mission_config.command.p2p_fine.long_forward_vx =
        get_parameter("p2p_fine_long_forward_vx").as_double();
    mission_config.command.p2p_fine.curve_yaw_threshold =
        get_parameter("p2p_fine_curve_yaw_threshold").as_double();
    mission_config.command.p2p_fine.turn_in_place_vx_max =
        get_parameter("p2p_fine_turn_in_place_vx_max").as_double();
    mission_config.command.p2p_fine.lateral_dominance_ratio =
        get_parameter("p2p_fine_lateral_dominance_ratio").as_double();
    mission_config.command.p2p_recovery.forward_deadband =
        get_parameter("p2p_recovery_forward_deadband").as_double();
    mission_config.command.p2p_recovery.lateral_deadband =
        get_parameter("p2p_recovery_lateral_deadband").as_double();
    mission_config.command.p2p_recovery.yaw_deadband =
        get_parameter("p2p_recovery_yaw_deadband").as_double();
    mission_config.command.p2p_recovery.long_forward_vx =
        get_parameter("p2p_recovery_long_forward_vx").as_double();
    mission_config.command.p2p_recovery.curve_yaw_threshold =
        get_parameter("p2p_recovery_curve_yaw_threshold").as_double();
    mission_config.command.p2p_recovery.turn_in_place_vx_max =
        get_parameter("p2p_recovery_turn_in_place_vx_max").as_double();
    mission_config.command.p2p_recovery.lateral_dominance_ratio =
        get_parameter("p2p_recovery_lateral_dominance_ratio").as_double();
    mission_config.line_observation_dt =
        1.0 / std::max(inference_hz_, 1e-6);
    mission_config.enable_ball =
        algorithm_mode_ == "ball" ||
        (algorithm_mode_ == "all" && enable_ball_controller_);
    mission_config.enable_hurdle =
        algorithm_mode_ == "hurdle" ||
        (algorithm_mode_ == "all" && enable_hurdle_controller_);
    mission_config.enable_goal =
        algorithm_mode_ == "goal" ||
        (algorithm_mode_ == "all" && enable_goal_controller_);
    // Goal 단독 실행만 공 집기 과정을 생략한다. 통합 모드는 반드시 Ball
    // 성공 결과를 통해서만 has_ball이 켜진다.
    mission_config.initial_has_ball = algorithm_mode_ == "goal";
    mission_controller_ =
        std::make_unique<vision_core::MissionController>(mission_config);

    camera_intrinsics_ = {fx, fy, cx, cy};

    // ----------------------------
    // ROS Pub/Sub
    // ----------------------------
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(), std::bind(&LinePerceptionNode::OnImage, this, std::placeholders::_1));
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic_, rclcpp::SensorDataQoS(), std::bind(&LinePerceptionNode::OnDepth, this, std::placeholders::_1));
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, rclcpp::SensorDataQoS(),
        std::bind(&LinePerceptionNode::OnCameraInfo, this, std::placeholders::_1));

    // 보정을 끄거나 zero-IMU 시험을 선택하면 입력 구독 자체를 만들지 않는다.
    if (use_imu_rectification_ && !assume_zero_imu_) {
      auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(10));
      imu_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
      imu_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
      imu_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
          imu_topic_, imu_qos,
          std::bind(&LinePerceptionNode::OnImu, this,
                    std::placeholders::_1));
    }

    prev_cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        prev_cmd_topic_, rclcpp::QoS(5), std::bind(&LinePerceptionNode::OnPrevCmd, this, std::placeholders::_1));

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_topic_, rclcpp::QoS(10));
    auto command_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    action_cmd_pub_ = create_publisher<vision::msg::ActionCommand>(action_cmd_topic_, command_qos);
    camera_cmd_pub_ = create_publisher<vision::msg::CameraCommand>(camera_cmd_topic_, command_qos);
    action_status_sub_ = create_subscription<vision::msg::CommandStatus>(
        action_status_topic_, command_qos,
        std::bind(&LinePerceptionNode::OnActionStatus, this, std::placeholders::_1));
    camera_status_sub_ = create_subscription<vision::msg::CommandStatus>(
        camera_status_topic_, command_qos,
        std::bind(&LinePerceptionNode::OnCameraStatus, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "vision node started.");
    RCLCPP_INFO(get_logger(), "  image_topic    : %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  depth_topic    : %s", depth_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  camera_info    : %s", camera_info_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  imu_topic      : %s", imu_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  prev_cmd_topic : %s", prev_cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  cmd_topic      : %s", cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  action_cmd     : %s", action_cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  action_status  : %s", action_status_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  camera_cmd     : %s", camera_cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  camera_status  : %s", camera_status_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  locomotion     : %s", locomotion_backend_.c_str());
    RCLCPP_INFO(get_logger(), "  engine_path    : %s", engine_path_.c_str());
    RCLCPP_INFO(get_logger(), "  line_class_id  : %d", line_class_id_);
    RCLCPP_INFO(get_logger(), "  ball_class_id  : %d", ball_class_id_);
    RCLCPP_INFO(get_logger(), "  goal_class_id  : %d", goal_class_id_);
    RCLCPP_INFO(get_logger(), "  backboard_cls  : %d", backboard_class_id_);
    RCLCPP_INFO(get_logger(), "  hurdle_class_id: %d", hurdle_class_id_);
    RCLCPP_INFO(get_logger(), "  ball_conf_thres: %.2f", ball_conf_thres_);
    RCLCPP_INFO(get_logger(), "  goal_conf_thres: %.2f", goal_conf_thres_);
    RCLCPP_INFO(get_logger(), "  backboard_conf : %.2f", backboard_conf_thres_);
    RCLCPP_INFO(get_logger(), "  hurdle_conf_thr: %.2f", hurdle_conf_thres_);
    RCLCPP_INFO(get_logger(), "  inference_hz   : %.2f", inference_hz_);
    RCLCPP_INFO(get_logger(), "  imu_rectify    : %s", use_imu_rectification_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  zero_imu       : %s", assume_zero_imu_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  rule_ctrl      : %s", enable_rule_controller_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  ball_ctrl      : %s", enable_ball_controller_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  hurdle_ctrl    : %s", enable_hurdle_controller_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  goal_ctrl      : %s", enable_goal_controller_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  algorithm_mode : %s", algorithm_mode_.c_str());
    if (ball_conf_thres_ < conf_thres_) {
      RCLCPP_WARN(get_logger(),
                  "ball_conf_thres(%.2f) < conf_thres(%.2f), but YOLO "
                  "engine output is already filtered by conf_thres.",
                  ball_conf_thres_, conf_thres_);
    }
    if (goal_conf_thres_ < conf_thres_) {
      RCLCPP_WARN(get_logger(),
                  "goal_conf_thres(%.2f) < conf_thres(%.2f), but YOLO "
                  "engine output is already filtered by conf_thres.",
                  goal_conf_thres_, conf_thres_);
    }
    if (backboard_conf_thres_ < conf_thres_) {
      RCLCPP_WARN(get_logger(),
                  "backboard_conf_thres(%.2f) < conf_thres(%.2f), but YOLO "
                  "engine output is already filtered by conf_thres.",
                  backboard_conf_thres_, conf_thres_);
    }
    if (hurdle_conf_thres_ < conf_thres_) {
      RCLCPP_WARN(get_logger(),
                  "hurdle_conf_thres(%.2f) < conf_thres(%.2f), but YOLO "
                  "engine output is already filtered by conf_thres.",
                  hurdle_conf_thres_, conf_thres_);
    }

    RCLCPP_INFO(get_logger(), "[SETUP] image subscription created (SensorDataQoS)");
    RCLCPP_INFO(get_logger(), "[SETUP] imu subscription created");
    RCLCPP_INFO(get_logger(), "[SETUP] cmd publisher created");

    if (show_debug_view_ || show_yolo_debug_view_) {
      try {
        if (show_debug_view_) {
          cv::namedWindow(kDebugWindowName, cv::WINDOW_NORMAL);
          cv::namedWindow(kStateWindowName, cv::WINDOW_NORMAL);
        }
        if (show_yolo_debug_view_) {
          cv::namedWindow(kYoloDebugWindowName, cv::WINDOW_NORMAL);
        }
        RCLCPP_INFO(get_logger(),
                    "[SETUP] confirmed_debug=%d yolo_debug=%d",
                    show_debug_view_ ? 1 : 0,
                    show_yolo_debug_view_ ? 1 : 0);
      } catch (const cv::Exception &e) {
        show_debug_view_ = false;
        show_yolo_debug_view_ = false;
        RCLCPP_WARN(get_logger(), "[SETUP] debug window disabled: %s", e.what());
      }
    }

    // ---- heartbeat(1Hz) ----
    hb_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {
      RCLCPP_INFO(get_logger(), "[HB] frames=%zu imu=%zu pub=%zu (last_img_stamp=%.9f)", frames_count_, imu_count_,
                  pub_count_, last_img_stamp_sec_);
    });

    if (inference_hz_ > 0.0) {
      const auto period =
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / inference_hz_));
      inference_timer_ = create_wall_timer(period, [this]() { ProcessLatestImage(); });
    }

    // ---- PERF(1초 갱신) 초기화: "오버레이만" ----
    last_report_time_ = this->get_clock()->now();
    perf_frame_count_ = 0;
    perf_infer_time_sec_ = 0.0;
    perf_loop_time_sec_ = 0.0;
    perf_text_ = "INFER: -- ms | LOOP: -- ms | FPS: --";
  }

  ~LinePerceptionNode() override {
    if (show_debug_view_ || show_yolo_debug_view_) {
      try {
        if (show_debug_view_) {
          cv::destroyWindow(kDebugWindowName);
          cv::destroyWindow(kStateWindowName);
        }
        if (show_yolo_debug_view_) {
          cv::destroyWindow(kYoloDebugWindowName);
        }
      } catch (...) {
      }
    }
  }

private:
  static constexpr const char *kDebugWindowName = "Vision Confirmed Debug";
  static constexpr const char *kStateWindowName = "Vision State Debug";
  static constexpr const char *kYoloDebugWindowName = "YOLO Detection Debug";
  static double Deg2Rad(const double deg) {
    return deg * M_PI / 180.0;
  }

  const char *ClassLabel(int class_id) const {
    if (class_id == line_class_id_) {
      return "line";
    }
    if (class_id == ball_class_id_) {
      return "ball";
    }
    if (class_id == goal_class_id_) {
      return "goal";
    }
    if (class_id == backboard_class_id_) {
      return "backboard";
    }
    if (class_id == hurdle_class_id_) {
      return "hurdle";
    }
    return "unknown";
  }

  static const char *ActionName(vision_core::MissionAction action) {
    using vision_core::MissionAction;
    switch (action) {
    case MissionAction::kNone: return "NONE";
    case MissionAction::kStepForwardHalf: return "STEP_FORWARD_HALF";
    case MissionAction::kStepForward: return "STEP_FORWARD";
    case MissionAction::kStepBackward: return "STEP_BACKWARD";
    case MissionAction::kStepLeft: return "STEP_LEFT";
    case MissionAction::kStepRight: return "STEP_RIGHT";
    case MissionAction::kTurnLeft: return "TURN_LEFT";
    case MissionAction::kTurnRight: return "TURN_RIGHT";
    case MissionAction::kPickupBall: return "PICKUP_BALL";
    case MissionAction::kStandUp: return "STAND_UP";
    case MissionAction::kHurdleContactWalk: return "HURDLE_CONTACT_WALK";
    case MissionAction::kCrossHurdle: return "CROSS_HURDLE";
    case MissionAction::kShoot: return "SHOOT";
    case MissionAction::kVerifyPickup: return "VERIFY_PICKUP";
    case MissionAction::kFineAdjustHold: return "FINE_ADJUST_HOLD";
    case MissionAction::kWalkForwardTwo: return "WALK_FORWARD_TWO";
    case MissionAction::kWalkForwardLeftTwo: return "WALK_FORWARD_LEFT_TWO";
    case MissionAction::kWalkForwardRightTwo: return "WALK_FORWARD_RIGHT_TWO";
    case MissionAction::kWalkForwardSix: return "WALK_FORWARD_SIX";
    case MissionAction::kWalkForwardLeftSix: return "WALK_FORWARD_LEFT_SIX";
    case MissionAction::kWalkForwardRightSix: return "WALK_FORWARD_RIGHT_SIX";
    case MissionAction::kWalkBackwardTwo: return "WALK_BACKWARD_TWO";
    case MissionAction::kWalkLeftTwo: return "WALK_LEFT_TWO";
    case MissionAction::kWalkRightTwo: return "WALK_RIGHT_TWO";
    case MissionAction::kTurnLeftInPlace: return "TURN_LEFT_IN_PLACE";
    case MissionAction::kTurnRightInPlace: return "TURN_RIGHT_IN_PLACE";
    }
    return "UNKNOWN";
  }

  bool ShouldDrawDebugView() const {
    return show_debug_view_ || show_yolo_debug_view_;
  }
  static double Clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
  }

  static double StampSeconds(const std_msgs::msg::Header &header) {
    return static_cast<double>(header.stamp.sec) + static_cast<double>(header.stamp.nanosec) * 1e-9;
  }

  static std::optional<double> MedianDepthMeters(const cv::Mat &depth, int center_u, int center_v) {
    if (depth.empty())
      return std::nullopt;
    std::vector<double> values;
    values.reserve(9);
    for (int dv = -1; dv <= 1; ++dv) {
      for (int du = -1; du <= 1; ++du) {
        const int u = center_u + du;
        const int v = center_v + dv;
        if (u < 0 || u >= depth.cols || v < 0 || v >= depth.rows)
          continue;
        double meters = 0.0;
        if (depth.type() == CV_16UC1) {
          meters = static_cast<double>(depth.at<std::uint16_t>(v, u)) * 0.001;
        } else if (depth.type() == CV_32FC1) {
          meters = static_cast<double>(depth.at<float>(v, u));
        } else {
          return std::nullopt;
        }
        if (std::isfinite(meters) && meters > 0.05 && meters < 10.0) {
          values.push_back(meters);
        }
      }
    }
    if (values.empty())
      return std::nullopt;
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
  }

  std::vector<vision_core::PerceptionDetection> BuildPerceptionDetections(
      const std::vector<Detection> &detections, const cv::Mat &depth) const {
    std::vector<vision_core::PerceptionDetection> output;
    output.reserve(detections.size());
    for (const auto &detection : detections) {
      vision_core::PerceptionDetection observation;
      observation.detection = {
          {static_cast<double>(detection.box.x),
           static_cast<double>(detection.box.y),
           static_cast<double>(detection.box.width),
           static_cast<double>(detection.box.height)},
          detection.confidence, detection.class_id};
      // 모든 backboard 후보에 표본을 붙인다. 최종 후보 선택과 pose 계산은
      // core에서 한 번만 수행한다.
      if (detection.class_id == backboard_class_id_ && !depth.empty()) {
        const int center_u = static_cast<int>(std::round(
            static_cast<double>(detection.box.x) +
            0.5 * static_cast<double>(detection.box.width)));
        const int left_u = detection.box.x;
        const int right_u = detection.box.x + detection.box.width - 1;
        const int middle_v = static_cast<int>(std::round(
            static_cast<double>(detection.box.y) +
            0.5 * static_cast<double>(detection.box.height)));
        observation.center_depth_m =
            MedianDepthMeters(depth, center_u, middle_v);
        observation.left_depth_m =
            MedianDepthMeters(depth, left_u, middle_v);
        observation.right_depth_m =
            MedianDepthMeters(depth, right_u, middle_v);
      }
      output.push_back(observation);
    }
    return output;
  }

  void OnPrevCmd(const geometry_msgs::msg::Twist::SharedPtr msg) {
    ++prev_cmd_count_;
    vx_prev_ = Clamp(static_cast<double>(msg->linear.x), vx_prev_min_, vx_prev_max_);
    wz_prev_ = Clamp(static_cast<double>(msg->angular.z), wz_prev_min_, wz_prev_max_);
  }

  void UpdateSimulatedDecisionFeedback(double now_sec) {
    if (!simulate_decision_ || simulated_action_id_ == 0) return;
    action_delivery_feedback_.action_id = simulated_action_id_;
    action_delivery_feedback_.acknowledged = true;
    if (now_sec - simulated_action_start_sec_ <
        std::max(0.0, simulate_action_duration_sec_)) {
      return;
    }
    action_delivery_feedback_.done = true;
    if (last_action_category_ == vision_core::ActionCategory::kMission) {
      mission_action_active_ = false;
    }
    simulated_action_id_ = 0;
  }

  void OnActionStatus(const vision::msg::CommandStatus::SharedPtr msg) {
    if (msg->command_id == 0 || msg->command_id != last_action_id_) return;
    action_delivery_feedback_.action_id = msg->command_id;
    if (msg->status == vision::msg::CommandStatus::ACK) {
      action_delivery_feedback_.acknowledged = true;
    } else if (msg->status == vision::msg::CommandStatus::DONE) {
      action_delivery_feedback_.acknowledged = true;
      action_delivery_feedback_.done = true;
      if (last_action_category_ == vision_core::ActionCategory::kMission) {
        mission_action_active_ = false;
      }
    }
  }

  void OnCameraStatus(const vision::msg::CommandStatus::SharedPtr msg) {
    if (msg->command_id == 0 || msg->command_id != pending_camera_id_) return;
    if (msg->status == vision::msg::CommandStatus::ACK) {
      camera_acknowledged_ = true;
      return;
    }
    if (msg->status != vision::msg::CommandStatus::DONE) return;
    switch (pending_camera_request_) {
    case vision_core::CameraRequest::kDown:
      camera_feedback_ = {vision_core::CameraMode::kDown, true};
      break;
    case vision_core::CameraRequest::kForward:
      camera_feedback_ = {vision_core::CameraMode::kForward, true};
      break;
    case vision_core::CameraRequest::kGoal:
      camera_feedback_ = {vision_core::CameraMode::kGoal, true};
      break;
    case vision_core::CameraRequest::kNone:
      break;
    }
    pending_camera_id_ = 0;
    pending_camera_request_ = vision_core::CameraRequest::kNone;
    camera_acknowledged_ = false;
  }

  void PublishActionCommand(const vision_core::ControlCommand &command) {
    if (!enable_command_transport_ ||
        command.command_type != vision_core::CommandType::kAction) {
      return;
    }
    vision::msg::ActionCommand message;
    message.action_id = command.action_id;
    message.mission = static_cast<std::uint8_t>(command.mission);
    message.action = static_cast<std::uint16_t>(command.action);
    const double target_yaw_deg =
        command.action_yaw_rad * 180.0 / M_PI;
    message.target_yaw_deg = static_cast<std::int16_t>(std::lround(
        Clamp(target_yaw_deg, -180.0, 180.0)));
    action_cmd_pub_->publish(message);
    last_action_id_ = command.action_id;
    last_action_category_ = command.action_category;
    if (simulate_decision_ && simulated_action_id_ != command.action_id) {
      simulated_action_id_ = command.action_id;
      simulated_action_start_sec_ = this->get_clock()->now().seconds();
    }
    if (command.action_category == vision_core::ActionCategory::kMission) {
      mission_action_active_ = true;
    }
  }

  void PublishCameraCommand(vision_core::CameraRequest request) {
    if (request == vision_core::CameraRequest::kNone) {
      return;
    }
    if (simulate_camera_feedback_) {
      switch (request) {
      case vision_core::CameraRequest::kDown:
        camera_feedback_ = {vision_core::CameraMode::kDown, true};
        break;
      case vision_core::CameraRequest::kForward:
        camera_feedback_ = {vision_core::CameraMode::kForward, true};
        break;
      case vision_core::CameraRequest::kGoal:
        camera_feedback_ = {vision_core::CameraMode::kGoal, true};
        break;
      case vision_core::CameraRequest::kNone:
        break;
      }
      return;
    }
    if (!enable_command_transport_) return;
    if (pending_camera_id_ == 0 || pending_camera_request_ != request) {
      pending_camera_id_ = next_camera_id_++;
      pending_camera_request_ = request;
      camera_acknowledged_ = false;
      camera_feedback_.settled = false;
      camera_feedback_.actual_mode = vision_core::CameraMode::kTransition;
    }
    if (camera_acknowledged_) return;
    vision::msg::CameraCommand message;
    message.command_id = pending_camera_id_;
    message.request = static_cast<std::uint8_t>(pending_camera_request_);
    camera_cmd_pub_->publish(message);
  }

  void OnImu(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
    ++imu_count_;

    std::lock_guard<std::mutex> lock(imu_mutex_);

    double r = static_cast<double>(msg->vector.x);
    double p = static_cast<double>(msg->vector.y);

    r = std::max(-imu_abs_limit_rad_, std::min(imu_abs_limit_rad_, r));
    p = std::max(-imu_abs_limit_rad_, std::min(imu_abs_limit_rad_, p));

    last_roll_rad_ = r;
    last_pitch_rad_ = p;

    imu_ready_ = true;

    RCLCPP_DEBUG(get_logger(), "[IMU] recv roll=%.5f pitch=%.5f (clamped) | imu_ready=%d", last_roll_rad_,
                 last_pitch_rad_, imu_ready_ ? 1 : 0);
  }

  void DrawYoloDetections(cv::Mat &vis,
                          const std::vector<Detection> &dets) {
    for (const auto &det : dets) {
      if (det.confidence < conf_thres_) {
        continue;
      }

      const cv::Rect box = det.box;
      cv::Rect clipped = box & cv::Rect(0, 0, vis.cols, vis.rows);
      if (clipped.width <= 0 || clipped.height <= 0) {
        continue;
      }

      const int cx = clipped.x + clipped.width / 2;
      const int cy = clipped.y + clipped.height / 2;

      cv::Scalar box_color(160, 160, 160);
      cv::Scalar center_color(80, 80, 80);
      if (det.class_id == line_class_id_) {
        box_color = cv::Scalar(0, 255, 0);
        center_color = cv::Scalar(0, 0, 255);
      } else if (det.class_id == ball_class_id_) {
        box_color = cv::Scalar(0, 165, 255);
        center_color = cv::Scalar(255, 0, 0);
      } else if (det.class_id == goal_class_id_) {
        box_color = cv::Scalar(255, 255, 0);
        center_color = cv::Scalar(0, 255, 255);
      } else if (det.class_id == backboard_class_id_) {
        box_color = cv::Scalar(255, 128, 0);
        center_color = cv::Scalar(0, 128, 255);
      } else if (det.class_id == hurdle_class_id_) {
        box_color = cv::Scalar(255, 0, 255);
        center_color = cv::Scalar(255, 255, 255);
      }

      cv::rectangle(vis, clipped, box_color, 2);
      cv::circle(vis, cv::Point(cx, cy), 3, center_color, -1);

      char buf[40];
      std::snprintf(buf, sizeof(buf), "%s %.2f", ClassLabel(det.class_id), det.confidence);

      int tx = clipped.x;
      int ty = clipped.y - 6;
      if (ty < 12)
        ty = clipped.y + 16;

      cv::putText(vis, buf, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX,
                  0.52, box_color, 1);
    }
  }

  void DrawConfirmedDetections(
      cv::Mat &vis, const std::vector<vision_core::Point2> &line_centers,
      bool line_confirmed,
      const std::optional<vision_core::ObjectTarget> &ball_target,
      bool ball_confirmed,
      const std::optional<vision_core::ObjectTarget> &backboard_target,
      bool backboard_confirmed,
      const std::optional<vision_core::ObjectTarget> &hurdle_target,
      bool hurdle_confirmed) {
    if (line_confirmed) {
      for (const auto &point : line_centers) {
        cv::circle(vis,
                   cv::Point(static_cast<int>(std::round(point.u)),
                             static_cast<int>(std::round(point.v))),
                   5, cv::Scalar(0, 255, 0), 2);
      }
    }

    const auto draw_target = [&vis](
                                 const std::optional<vision_core::ObjectTarget> &target,
                                 bool confirmed, const char *label,
                                 const cv::Scalar &color) {
      if (!confirmed || !target) return;
      const auto &box = target->box_px;
      const cv::Rect raw_box(static_cast<int>(std::round(box.x)),
                             static_cast<int>(std::round(box.y)),
                             static_cast<int>(std::round(box.width)),
                             static_cast<int>(std::round(box.height)));
      const cv::Rect clipped = raw_box & cv::Rect(0, 0, vis.cols, vis.rows);
      if (clipped.width <= 0 || clipped.height <= 0) return;
      cv::rectangle(vis, clipped, color, 2);
      const auto center = target->center_rectified
                              ? target->rectified_center_px
                              : target->center_px;
      cv::circle(vis,
                 cv::Point(static_cast<int>(std::round(center.u)),
                           static_cast<int>(std::round(center.v))),
                 4, color, cv::FILLED);
      const int text_y = clipped.y >= 14 ? clipped.y - 4 : clipped.y + 14;
      cv::putText(vis, label, cv::Point(clipped.x, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.60, color, 2);
    };

    draw_target(ball_target, ball_confirmed, "BALL",
                cv::Scalar(0, 165, 255));
    draw_target(backboard_target, backboard_confirmed, "BACKBOARD APPROACH",
                cv::Scalar(255, 128, 0));
    draw_target(hurdle_target, hurdle_confirmed, "HURDLE",
                cv::Scalar(255, 0, 255));
  }

  void DrawTargetSummary(cv::Mat &vis, const char *name, const std::optional<vision_core::ObjectTarget> &target, int y,
                         const cv::Scalar &color) {
    if (!target) {
      return;
    }

    const auto p = target->center_rectified ? target->rectified_center_px : target->center_px;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s: conf=%.2f u=%.1f v=%.1f h=%.0f area=%.0f", name, target->confidence, p.u, p.v,
                  target->height_px, target->area_px);
    cv::putText(vis, buf, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                0.52, color, 1);
  }

  // PERF: 1초에 1번만 문자열 갱신 (로그로는 절대 안 찍음)
  void UpdatePerfOverlayOncePerSecond(double infer_time_sec, double loop_time_sec) {
    const rclcpp::Time now = this->get_clock()->now();
    const rclcpp::Duration elapsed = now - last_report_time_;

    perf_frame_count_ += 1;
    perf_infer_time_sec_ += infer_time_sec;
    perf_loop_time_sec_ += loop_time_sec;

    if (elapsed.seconds() >= 1.0) {
      const int denom = std::max(1, perf_frame_count_);
      const double avg_infer_ms = (perf_infer_time_sec_ / static_cast<double>(denom)) * 1000.0;
      const double avg_loop_ms = (perf_loop_time_sec_ / static_cast<double>(denom)) * 1000.0;
      const double fps = static_cast<double>(perf_frame_count_) / elapsed.seconds();

      last_report_time_ = now;
      perf_frame_count_ = 0;
      perf_infer_time_sec_ = 0.0;
      perf_loop_time_sec_ = 0.0;

      char buf[128];
      std::snprintf(buf, sizeof(buf), "INFER: %.2fms | LOOP: %.2fms | FPS: %.2f", avg_infer_ms, avg_loop_ms, fps);
      perf_text_ = buf;
    }
  }

  void OnImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    ++frames_count_;
    last_img_stamp_sec_ =
        static_cast<double>(msg->header.stamp.sec) + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    if (inference_hz_ > 0.0) {
      std::lock_guard<std::mutex> lock(image_mutex_);
      latest_image_ = msg;
      return;
    }

    ProcessImage(msg);
  }

  void OnDepth(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_depth_ = msg;
  }

  void OnCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (msg->k[0] <= 0.0 || msg->k[4] <= 0.0)
      return;
    camera_intrinsics_ = {msg->k[0], msg->k[4], msg->k[2], msg->k[5]};
    camera_info_ready_ = true;
  }

  void ProcessLatestImage() {
    sensor_msgs::msg::Image::SharedPtr msg;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      msg = latest_image_;
      latest_image_.reset();
    }

    if (!msg) {
      return;
    }

    ProcessImage(msg);
  }

  void ProcessImage(const sensor_msgs::msg::Image::SharedPtr &msg) {
    const auto t0 = std::chrono::steady_clock::now();

    RCLCPP_DEBUG(get_logger(), "[STEP %zu] Process image", frames_count_);

    // 1) ROS Image -> cv::Mat(BGR8)
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(get_logger(), "[IMG] cv_bridge 예외: %s", e.what());
      return;
    }

    const cv::Mat &bgr = cv_ptr->image;
    if (bgr.empty()) {
      RCLCPP_WARN(get_logger(), "[IMG] 빈 이미지가 들어왔습니다.");
      return;
    }

    // 최신 aligned depth를 같은 색상 프레임에 대응시킨다. 단독 시험용
    // latest-value 결합이며, 시간 차가 크거나 해상도가 다르면 자세 계산에 쓰지 않는다.
    sensor_msgs::msg::Image::SharedPtr depth_msg;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      depth_msg = latest_depth_;
    }
    cv_bridge::CvImageConstPtr depth_cv_ptr;
    cv::Mat aligned_depth;
    if (depth_msg &&
        std::abs(StampSeconds(msg->header) - StampSeconds(depth_msg->header)) <= std::max(0.0, max_depth_age_sec_)) {
      try {
        depth_cv_ptr = cv_bridge::toCvShare(depth_msg);
        if (depth_cv_ptr->image.cols == bgr.cols && depth_cv_ptr->image.rows == bgr.rows) {
          aligned_depth = depth_cv_ptr->image;
        }
      } catch (const cv_bridge::Exception &e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "[DEPTH] cv_bridge exception: %s", e.what());
      }
    }

    // 2) IMU 최신값 스냅샷
    double roll = 0.0;
    double pitch = 0.0;
    bool imu_ready_snapshot = false;
    if (use_imu_rectification_ && !assume_zero_imu_) {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      roll = last_roll_rad_;
      pitch = last_pitch_rad_;
      imu_ready_snapshot = imu_ready_;
    } else if (use_imu_rectification_ && assume_zero_imu_) {
      imu_ready_snapshot = true;
    }
    RCLCPP_DEBUG(get_logger(), "[IMU] snapshot roll=%.5f pitch=%.5f | imu_ready=%d", roll, pitch,
                 imu_ready_snapshot ? 1 : 0);

    // 3) YOLO 추론
    std::vector<Detection> dets;

    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = yolo_->Infer(bgr, dets);
    const auto t2 = std::chrono::steady_clock::now();

    const auto dt_yolo_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (!ok) {
      RCLCPP_ERROR(get_logger(), "[YOLO] Infer failed -> skip remaining stages");
      return;
    }

    // 4) ROS/sensor 형식을 core의 raw perception 입력으로만 변환한다.
    const auto t3 = std::chrono::steady_clock::now();
    const double frame_now_sec = this->get_clock()->now().seconds();
    UpdateSimulatedDecisionFeedback(frame_now_sec);
    vision_core::PerceptionFrameInput perception_input;
    perception_input.detections =
        BuildPerceptionDetections(dets, aligned_depth);
    perception_input.intrinsics = camera_intrinsics_;
    perception_input.enable_imu_rectification = use_imu_rectification_;
    perception_input.imu_valid = imu_ready_snapshot;
    perception_input.roll_rad = roll;
    perception_input.pitch_rad = pitch;
    perception_input.previous_vx = vx_prev_;
    perception_input.previous_wz = wz_prev_;
    perception_input.image_width = bgr.cols;
    perception_input.image_height = bgr.rows;
    perception_input.now_sec = frame_now_sec;
    perception_input.camera_feedback = camera_feedback_;
    perception_input.command_transport_enabled = enable_command_transport_;
    perception_input.delivery_feedback = action_delivery_feedback_;
    const auto t4 = std::chrono::steady_clock::now();
    const auto dt_pts_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // 5) 전처리, 선택적 IMU 보정, 상태 전이, 최종 명령은 core에서 수행한다.
    const auto t5 = std::chrono::steady_clock::now();
    const auto perception_result =
        mission_controller_->StepPerception(perception_input);
    const auto t6 = std::chrono::steady_clock::now();
    const auto dt_rec_us = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

    const auto &prepared = perception_result.perception;
    const auto &object_targets = prepared.targets;
    const auto &ball_target = object_targets.ball;
    const auto &goal_target = object_targets.goal;
    const auto &backboard_target = object_targets.backboard;
    const auto &hurdle_target = object_targets.hurdle;
    const auto &goal_pose = prepared.goal_pose;
    const auto &backboard_center_depth_m =
        prepared.backboard_center_depth_m;
    const auto &backboard_left_depth_m =
        prepared.backboard_left_depth_m;
    const auto &backboard_right_depth_m =
        prepared.backboard_right_depth_m;

    // 6) core가 반환한 최종 명령을 ROS 토픽 형식으로 전달한다.
    const auto t7 = std::chrono::steady_clock::now();
    const auto &mission_result = perception_result.mission;
    const auto &control_command = mission_result.command;
    const auto &ball_cmd = mission_result.ball;
    const auto &hurdle_cmd = mission_result.hurdle;
    const auto &goal_cmd = mission_result.goal;
    const auto &feats = mission_result.line_features;
    const bool selected_ball =
        control_command.mission == vision_core::MissionType::kBall;
    const bool selected_hurdle =
        control_command.mission == vision_core::MissionType::kHurdle;
    const bool selected_goal =
        control_command.mission == vision_core::MissionType::kGoal;
    // core가 최종 P2P action과 함께 반환한 양자화 전 연속속도 의도다.
    // ROS wire action 메시지는 그대로 두고 디버그 화면에서만 표시한다.
    const vision_core::MotionCommand requested_motion =
        control_command.pre_p2p_motion;
    const char *selected_mode =
        selected_ball
            ? vision_core::BallController::ModeName(ball_cmd.mode)
            : (selected_hurdle
                   ? vision_core::HurdleController::ModeName(hurdle_cmd.mode)
                   : (selected_goal
                          ? vision_core::GoalController::ModeName(goal_cmd.mode)
                          : "LINE_RULE"));

    geometry_msgs::msg::Twist selected_cmd;
    selected_cmd.linear.x = control_command.velocity.vx;
    selected_cmd.linear.y = control_command.velocity.vy;
    selected_cmd.angular.z = control_command.velocity.wz;

    vx_prev_ = Clamp(static_cast<double>(selected_cmd.linear.x), vx_prev_min_, vx_prev_max_);
    wz_prev_ = Clamp(static_cast<double>(selected_cmd.angular.z), wz_prev_min_, wz_prev_max_);

    if (enable_rule_controller_ && locomotion_backend_ == "velocity") {
      cmd_pub_->publish(selected_cmd);
      ++pub_count_;
    }
    PublishActionCommand(control_command);
    PublishCameraCommand(control_command.camera_request);
    action_delivery_feedback_ = {};

    const auto t8 = std::chrono::steady_clock::now();
    const auto dt_feat_us = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();

    // ---- 시각화 ----
    if (ShouldDrawDebugView()) {
      try {
        if (show_yolo_debug_view_) {
          cv::Mat yolo_vis = bgr.clone();
          DrawYoloDetections(yolo_vis, dets);
          cv::imshow(kYoloDebugWindowName, yolo_vis);
        }

        if (show_debug_view_) {
          const bool line_confirmed =
              mission_result.line_computed && feats.n_visible > 0.0 &&
              feats.in_recovery <= 0.5;
          const bool ball_confirmed =
              ball_target && ball_cmd.tracked.stable &&
              ball_cmd.tracked.visible;
          const bool backboard_confirmed =
              backboard_target && goal_cmd.tracked.stable &&
              goal_cmd.tracked.visible;
          const bool hurdle_confirmed =
              hurdle_target && hurdle_cmd.tracked.stable &&
              hurdle_cmd.tracked.visible;
          const bool pose_confirmed =
              backboard_confirmed && goal_cmd.pose.stable &&
              goal_cmd.pose.visible && goal_pose.valid;

          cv::Mat vis = bgr.clone();
          DrawConfirmedDetections(
              vis, prepared.line_centers, line_confirmed, ball_target,
              ball_confirmed, backboard_target, backboard_confirmed,
              hurdle_target, hurdle_confirmed);

          if (pose_confirmed) {
            const auto &box = backboard_target->box_px;
            const int left_x = static_cast<int>(std::round(box.x));
            const int right_x =
                static_cast<int>(std::round(box.x + box.width - 1.0));
            const int y =
                static_cast<int>(std::round(box.y + 0.5 * box.height));
            const cv::Rect image_rect(0, 0, vis.cols, vis.rows);
            cv::rectangle(vis, cv::Rect(left_x - 1, y - 1, 3, 3) & image_rect,
                          cv::Scalar(0, 0, 255), cv::FILLED);
            cv::rectangle(vis,
                          cv::Rect(right_x - 1, y - 1, 3, 3) & image_rect,
                          cv::Scalar(0, 0, 255), cv::FILLED);
          }

          cv::putText(vis, perf_text_, cv::Point(10, 24),
                      cv::FONT_HERSHEY_SIMPLEX, 0.70,
                      cv::Scalar(0, 255, 0), 2);
          char overlay[256];
          const bool show_p2p_action = locomotion_backend_ == "p2p";
          if (show_p2p_action) {
            std::snprintf(
                overlay, sizeof(overlay),
                "MODE: %s | ACTION: %u %s", selected_mode,
                static_cast<unsigned int>(control_command.action),
                ActionName(control_command.action));
          } else {
            std::snprintf(overlay, sizeof(overlay),
                          "MODE: %s | CMD: vx=%+.3f vy=%+.3f wz=%+.3f",
                          selected_mode, selected_cmd.linear.x,
                          selected_cmd.linear.y, selected_cmd.angular.z);
          }
          cv::putText(vis, overlay, cv::Point(10, 48),
                      cv::FONT_HERSHEY_SIMPLEX,
                      show_p2p_action ? 0.55 : 0.70,
                      cv::Scalar(0, 200, 255), 2);
          if (show_p2p_action) {
            std::snprintf(
                overlay, sizeof(overlay),
                "PRE-P2P: vx=%+.3f vy=%+.3f wz=%+.3f",
                requested_motion.vx, requested_motion.vy,
                requested_motion.wz);
            cv::putText(vis, overlay, cv::Point(10, 70),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 200, 255), 2);
          }

          cv::Mat state_panel(360, 1100, CV_8UC3,
                              cv::Scalar(20, 20, 20));
          int state_y = 24;
          const auto draw_state = [&state_panel, &state_y](
                                      const char *text,
                                      const cv::Scalar &color) {
            cv::putText(state_panel, text, cv::Point(10, state_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.52, color, 1);
            state_y += 28;
          };

          char buf[256];
          if (!use_imu_rectification_) {
            std::snprintf(buf, sizeof(buf), "IMU: DISABLED");
          } else if (imu_ready_snapshot) {
            std::snprintf(buf, sizeof(buf),
                          "IMU: roll=%+.1fdeg pitch=%+.1fdeg",
                          roll * 180.0 / M_PI, pitch * 180.0 / M_PI);
          } else {
            std::snprintf(buf, sizeof(buf), "IMU: NOT READY");
          }
          draw_state(buf, cv::Scalar(255, 255, 255));

          if (line_confirmed) {
            std::snprintf(buf, sizeof(buf),
                          "LINE STATE: active=%d confirmed=1 dots=%.0f u_err=%+.3f slope=%+.3f",
                          selected_ball || selected_hurdle || selected_goal ? 0 : 1,
                          feats.n_visible, feats.u_err_ctrl, feats.slope);
          } else {
            std::snprintf(buf, sizeof(buf),
                          "LINE STATE: active=%d confirmed=0 dots=-- u_err=-- slope=--",
                          selected_ball || selected_hurdle || selected_goal ? 0 : 1);
          }
          draw_state(buf, cv::Scalar(0, 255, 0));

          if (ball_confirmed) {
            std::snprintf(buf, sizeof(buf),
                          "BALL STATE: active=%d mode=%s stable=1 visible=1 u=%.2f v=%.2f h=%.2f",
                          selected_ball ? 1 : 0,
                          vision_core::BallController::ModeName(ball_cmd.mode),
                          ball_cmd.tracked.u_norm, ball_cmd.tracked.v_norm,
                          ball_cmd.tracked.h_norm);
          } else {
            std::snprintf(buf, sizeof(buf),
                          "BALL STATE: active=%d mode=%s stable=%d visible=%d u=-- v=-- h=--",
                          selected_ball ? 1 : 0,
                          vision_core::BallController::ModeName(ball_cmd.mode),
                          ball_cmd.tracked.stable ? 1 : 0,
                          ball_cmd.tracked.visible ? 1 : 0);
          }
          draw_state(buf, cv::Scalar(0, 165, 255));
          if (ball_confirmed) {
            DrawTargetSummary(state_panel, "BALL BBOX", ball_target, state_y,
                              cv::Scalar(0, 165, 255));
          } else {
            draw_state("BALL BBOX: --", cv::Scalar(0, 165, 255));
          }
          if (ball_confirmed) state_y += 28;

          if (pose_confirmed) {
            std::snprintf(
                buf, sizeof(buf),
                "GOAL POSE: valid=1 x=%+.3fm z=%.3fm backboard_yaw=%+.1fdeg",
                goal_cmd.pose.x_m, goal_cmd.pose.z_m,
                goal_cmd.pose.yaw_rad * 180.0 / M_PI);
          } else {
            std::snprintf(buf, sizeof(buf),
                          "GOAL POSE: valid=0 x=-- z=-- backboard_yaw=--");
          }
          draw_state(buf, cv::Scalar(0, 255, 255));

          const bool shoot_yaw_valid =
              pose_confirmed ||
              (selected_goal && goal_cmd.mode == vision_core::GoalMode::kShoot);
          if (shoot_yaw_valid) {
            std::snprintf(buf, sizeof(buf),
                          "SHOOT ROTATION YAW: %+.1fdeg",
                          goal_cmd.shoot_yaw_rad * 180.0 / M_PI);
          } else {
            std::snprintf(buf, sizeof(buf), "SHOOT ROTATION YAW: --");
          }
          draw_state(buf, cv::Scalar(0, 200, 255));

          if (backboard_confirmed) {
            std::snprintf(
                buf, sizeof(buf),
                "BACKBOARD APPROACH: active=%d mode=%s stable=1 visible=1 u=%.2f v=%.2f",
                selected_goal ? 1 : 0,
                vision_core::GoalController::ModeName(goal_cmd.mode),
                goal_cmd.tracked.u_norm, goal_cmd.tracked.v_norm);
          } else {
            std::snprintf(
                buf, sizeof(buf),
                "BACKBOARD APPROACH: active=%d mode=%s stable=%d visible=%d u=-- v=--",
                selected_goal ? 1 : 0,
                vision_core::GoalController::ModeName(goal_cmd.mode),
                goal_cmd.tracked.stable ? 1 : 0,
                goal_cmd.tracked.visible ? 1 : 0);
          }
          draw_state(buf, cv::Scalar(255, 128, 0));

          const auto depth_text = [](const std::optional<double> &depth) {
            if (!depth || !std::isfinite(*depth)) return std::string("--");
            char value[32];
            std::snprintf(value, sizeof(value), "%.3fm", *depth);
            return std::string(value);
          };
          const std::string center_depth =
              backboard_confirmed
                  ? depth_text(backboard_center_depth_m)
                  : "--";
          const std::string left_depth =
              backboard_confirmed ? depth_text(backboard_left_depth_m) : "--";
          const std::string right_depth =
              backboard_confirmed ? depth_text(backboard_right_depth_m) : "--";
          std::snprintf(buf, sizeof(buf),
                        "BACKBOARD DEPTH: center=%s left=%s right=%s",
                        center_depth.c_str(), left_depth.c_str(),
                        right_depth.c_str());
          draw_state(buf, cv::Scalar(255, 128, 0));

          if (hurdle_confirmed) {
            std::snprintf(buf, sizeof(buf),
                          "HURDLE STATE: active=%d mode=%s stable=1 visible=1",
                          selected_hurdle ? 1 : 0,
                          vision_core::HurdleController::ModeName(
                              hurdle_cmd.mode));
          } else {
            std::snprintf(buf, sizeof(buf),
                          "HURDLE STATE: active=%d mode=%s stable=%d visible=%d",
                          selected_hurdle ? 1 : 0,
                          vision_core::HurdleController::ModeName(
                              hurdle_cmd.mode),
                          hurdle_cmd.tracked.stable ? 1 : 0,
                          hurdle_cmd.tracked.visible ? 1 : 0);
          }
          draw_state(buf, cv::Scalar(255, 0, 255));
          if (hurdle_confirmed) {
            DrawTargetSummary(state_panel, "HURDLE BBOX", hurdle_target,
                              state_y, cv::Scalar(255, 0, 255));
          } else {
            draw_state("HURDLE BBOX: --", cv::Scalar(255, 0, 255));
          }

          cv::imshow(kDebugWindowName, vis);
          cv::imshow(kStateWindowName, state_panel);
        }
        cv::waitKey(1);
      } catch (const cv::Exception &e) {
        show_debug_view_ = false;
        show_yolo_debug_view_ = false;
        RCLCPP_WARN(get_logger(), "[VIEW] debug window disabled: %s", e.what());
      }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const auto dt_loop_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t0).count();

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "[TIME] yolo=%.2fms pts=%.2fms rect=%.2fms "
                         "feat=%.2fms infer=%.2fms loop=%.2fms n=%.0f rec=%.0f "
                         "ball=%d goal=%d backboard=%d hurdle=%d mode=%s "
                         "pose=%d x=%+.3f z=%.3f yaw=%+.1fdeg vx=%.3f vy=%.3f wz=%.3f",
                         static_cast<double>(dt_yolo_us) / 1000.0, static_cast<double>(dt_pts_us) / 1000.0,
                         static_cast<double>(dt_rec_us) / 1000.0, static_cast<double>(dt_feat_us) / 1000.0,
                         static_cast<double>(dt_yolo_us) / 1000.0, static_cast<double>(dt_loop_us) / 1000.0,
                         feats.n_visible, feats.in_recovery, ball_target ? 1 : 0, goal_target ? 1 : 0,
                         backboard_target ? 1 : 0, hurdle_target ? 1 : 0, selected_mode, goal_pose.valid ? 1 : 0,
                         goal_pose.x_m, goal_pose.z_m, goal_pose.yaw_rad * 180.0 / M_PI, selected_cmd.linear.x,
                         selected_cmd.linear.y, selected_cmd.angular.z);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "[CMD] %s -> %s | vx=%.3f vy=%.3f wz=%.3f | line_n=%.0f ball_stable=%d request_cam_down=%d",
                         selected_goal ? "GOAL" : (selected_hurdle ? "HURDLE" : (selected_ball ? "BALL" : "LINE")),
                         cmd_topic_.c_str(), selected_cmd.linear.x, selected_cmd.linear.y, selected_cmd.angular.z,
                         feats.n_visible, ball_cmd.tracked.stable ? 1 : 0,
                         ball_cmd.camera_request == vision_core::CameraRequest::kDown ? 1 : 0);

    // ===== PERF overlay 업데이트(1초에 1번만 문자열 갱신) =====
    UpdatePerfOverlayOncePerSecond(static_cast<double>(dt_yolo_us) * 1e-6, static_cast<double>(dt_loop_us) * 1e-6);
  }

private:
  // ROS Sub/Pub
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr prev_cmd_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<vision::msg::ActionCommand>::SharedPtr action_cmd_pub_;
  rclcpp::Publisher<vision::msg::CameraCommand>::SharedPtr camera_cmd_pub_;
  rclcpp::Subscription<vision::msg::CommandStatus>::SharedPtr action_status_sub_;
  rclcpp::Subscription<vision::msg::CommandStatus>::SharedPtr camera_status_sub_;
  rclcpp::TimerBase::SharedPtr hb_timer_;
  rclcpp::TimerBase::SharedPtr inference_timer_;

  // 모듈
  std::unique_ptr<YoloTrtEngine> yolo_;
  std::unique_ptr<vision_core::MissionController> mission_controller_;
  vision_core::Intrinsics camera_intrinsics_{};

  // IMU 최신 상태(latest)
  std::mutex imu_mutex_;
  bool imu_ready_{false};
  bool use_imu_rectification_{false};
  bool assume_zero_imu_{false};
  double last_roll_rad_{0.0};
  double last_pitch_rad_{0.0};
  double imu_abs_limit_rad_{Deg2Rad(45.0)};

  std::mutex image_mutex_;
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  sensor_msgs::msg::Image::SharedPtr latest_depth_;
  bool camera_info_ready_{false};
  double vx_prev_{0.0};
  double wz_prev_{0.0};
  double vx_prev_min_{0.0};
  double vx_prev_max_{1.2};
  double wz_prev_min_{-1.9};
  double wz_prev_max_{1.9};

  bool enable_rule_controller_{true};
  bool enable_ball_controller_{false};
  bool enable_hurdle_controller_{false};
  bool enable_goal_controller_{false};
  bool simulate_camera_feedback_{true};
  bool simulate_decision_{false};
  double simulate_action_duration_sec_{0.50};
  bool enable_command_transport_{true};
  vision_core::CommandDeliveryFeedback action_delivery_feedback_{};
  std::uint64_t last_action_id_{0};
  std::uint64_t simulated_action_id_{0};
  double simulated_action_start_sec_{0.0};
  vision_core::ActionCategory last_action_category_{
      vision_core::ActionCategory::kNone};
  bool mission_action_active_{false};
  vision_core::CameraFeedback camera_feedback_{};
  std::uint64_t next_camera_id_{1};
  std::uint64_t pending_camera_id_{0};
  vision_core::CameraRequest pending_camera_request_{
      vision_core::CameraRequest::kNone};
  bool camera_acknowledged_{false};
  double max_depth_age_sec_{0.20};
  std::string algorithm_mode_{"all"};
  std::string locomotion_backend_{"p2p"};

  // 디버그 카운터
  size_t frames_count_{0};
  size_t imu_count_{0};
  size_t pub_count_{0};
  size_t prev_cmd_count_{0};
  double last_img_stamp_sec_{0.0};

  // ---- PERF 오버레이(1초 갱신) ----
  rclcpp::Time last_report_time_{0, 0, RCL_ROS_TIME};
  int perf_frame_count_{0};
  double perf_infer_time_sec_{0.0};
  double perf_loop_time_sec_{0.0};
  std::string perf_text_;

  // ---- 시각화/필터 설정(멤버) ----
  int line_class_id_{0};
  int ball_class_id_{1};
  int goal_class_id_{2};
  int backboard_class_id_{3};
  int hurdle_class_id_{4};
  float conf_thres_{0.60f};
  float ball_conf_thres_{0.60f};
  float goal_conf_thres_{0.60f};
  float backboard_conf_thres_{0.60f};
  float hurdle_conf_thres_{0.60f};
  bool show_debug_view_{false};
  bool show_yolo_debug_view_{false};
  double inference_hz_{5.0};

  std::string image_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string imu_topic_;
  std::string prev_cmd_topic_;
  std::string cmd_topic_;
  std::string action_cmd_topic_;
  std::string action_status_topic_;
  std::string camera_cmd_topic_;
  std::string camera_status_topic_;
  std::string engine_path_;
};

} // namespace vision

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<vision::LinePerceptionNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("vision"), "치명적 오류: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
