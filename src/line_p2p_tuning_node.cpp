// 실제 로봇 LINE P2P O/H gain과 steering deadband 단발 튜닝 전용 노드.
// 운영용 line_perception_node와 실행 상태를 공유하지 않으며, YOLO adapter와
// vision_core::MissionController를 그대로 사용해 알고리즘 중복을 피한다.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/parameter_map.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <sys/utsname.h>

#include "vision/msg/action_command.hpp"
#include "vision/msg/command_status.hpp"
#include "vision/types.hpp"
#include "vision/yolo_trt_engine.hpp"
#include "vision_core/config_loader.hpp"
#include "vision_core/mission_controller.hpp"

namespace vision {
namespace {

bool IsJetsonTarget() {
  struct utsname info {};
  if (uname(&info) != 0) return false;
  const std::string machine(info.machine);
  return machine == "aarch64" || machine == "arm64";
}

std::string RuntimeConfigPath() {
  if (const char *path = std::getenv("YOLO26_RUNTIME_CONFIG");
      path != nullptr && path[0] != '\0') {
    return path;
  }
  return ament_index_cpp::get_package_share_directory("vision") +
         "/config/yolo26_runtime.yaml";
}

std::string ReadRuntimeString(const std::string &name) {
  try {
    const auto map = rclcpp::parameter_map_from_yaml_file(RuntimeConfigPath());
    for (const auto &node_name : {"/yolo26_runtime", "yolo26_runtime"}) {
      const auto node = map.find(node_name);
      if (node == map.end()) continue;
      for (const auto &parameter : node->second) {
        if (parameter.get_name() == name) return parameter.as_string();
      }
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[WARN] Failed to load YOLO runtime config: %s\n",
                 error.what());
  }
  return {};
}

std::string DefaultEnginePath() {
  if (const char *path = std::getenv("YOLO26_ENGINE_PATH");
      path != nullptr && path[0] != '\0') {
    return path;
  }
  return ReadRuntimeString(IsJetsonTarget() ? "jetson_engine_path"
                                            : "pc_engine_path");
}

} // namespace

class LineP2pTuningNode final : public rclcpp::Node {
public:
  LineP2pTuningNode() : rclcpp::Node("line_p2p_tuning_node") {
    auto config = vision_core::LoadDefaultAlgorithmConfig();

    declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
    declare_parameter<std::string>("camera_info_topic",
                                   "/camera/color/camera_info");
    declare_parameter<std::string>("imu_topic", "/camera/imu_tilt");
    declare_parameter<std::string>("action_cmd_topic",
                                   "/jandi_vision/action_cmd");
    declare_parameter<std::string>("action_status_topic",
                                   "/jandi_vision/action_status");
    declare_parameter<std::string>("engine_path", DefaultEnginePath());
    declare_parameter<int>("line_class_id", config.line_detection.class_id);
    declare_parameter<double>("conf_thres",
                              config.line_detection.confidence);
    declare_parameter<double>("fx", 600.0);
    declare_parameter<double>("fy", 600.0);
    declare_parameter<double>("cx", 320.0);
    declare_parameter<double>("cy", 240.0);
    declare_parameter<bool>("use_imu_rectification", false);
    declare_parameter<bool>("assume_zero_imu", false);
    declare_parameter<double>("imu_abs_limit_deg", 45.0);
    declare_parameter<double>("inference_hz", 15.0);
    declare_parameter<bool>("show_debug_view", true);
    declare_parameter<double>("line_tuning_observation_sec", 1.5);

    declare_parameter<int>("max_centers", config.line_features.max_centers);
    declare_parameter<double>("line_p2p_offset_gain",
                              config.line_p2p.offset_gain);
    declare_parameter<double>("line_p2p_heading_gain",
                              config.line_p2p.heading_gain);
    declare_parameter<double>("line_p2p_steering_deadband",
                              config.line_p2p.steering_deadband);

    image_topic_ = get_parameter("image_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();
    imu_topic_ = get_parameter("imu_topic").as_string();
    action_cmd_topic_ = get_parameter("action_cmd_topic").as_string();
    action_status_topic_ = get_parameter("action_status_topic").as_string();
    const std::string engine_path = get_parameter("engine_path").as_string();
    line_class_id_ = get_parameter("line_class_id").as_int();
    conf_thres_ = get_parameter("conf_thres").as_double();
    use_imu_rectification_ =
        get_parameter("use_imu_rectification").as_bool();
    assume_zero_imu_ = get_parameter("assume_zero_imu").as_bool();
    imu_abs_limit_rad_ =
        get_parameter("imu_abs_limit_deg").as_double() * M_PI / 180.0;
    inference_hz_ = get_parameter("inference_hz").as_double();
    show_debug_view_ = get_parameter("show_debug_view").as_bool();
    observation_sec_ = get_parameter("line_tuning_observation_sec").as_double();
    hold_sec_ = config.line_p2p.failure_observation_sec;

    if (engine_path.empty()) {
      throw std::runtime_error("engine_path is empty");
    }
    if (!std::isfinite(observation_sec_) || observation_sec_ <= 0.0) {
      throw std::invalid_argument("line_tuning_observation_sec must be > 0");
    }
    if (!std::isfinite(hold_sec_) || hold_sec_ < 0.0) {
      throw std::invalid_argument(
          "line_p2p.failure_observation_sec must be finite and >= 0");
    }

    config.enable_ball = false;
    config.enable_hurdle = false;
    config.enable_goal = false;
    config.initial_has_ball = false;
    config.line_detection.class_id = line_class_id_;
    config.line_detection.confidence = conf_thres_;
    config.line_features.max_centers = get_parameter("max_centers").as_int();
    config.line_features.image_center_u = get_parameter("cx").as_double();

    config.line_p2p.offset_gain =
        get_parameter("line_p2p_offset_gain").as_double();
    config.line_p2p.heading_gain =
        get_parameter("line_p2p_heading_gain").as_double();
    config.line_p2p.steering_deadband =
        get_parameter("line_p2p_steering_deadband").as_double();
    if (!std::isfinite(config.line_p2p.offset_gain) ||
        config.line_p2p.offset_gain < 0.0 ||
        !std::isfinite(config.line_p2p.heading_gain) ||
        config.line_p2p.heading_gain < 0.0 ||
        !std::isfinite(config.line_p2p.steering_deadband) ||
        config.line_p2p.steering_deadband < 0.0) {
      throw std::invalid_argument(
          "LINE P2P gains and steering deadband must be finite and >= 0");
    }
    line_p2p_config_ = config.line_p2p;

    controller_ = std::make_unique<vision_core::MissionController>(config);
    yolo_ = std::make_unique<YoloTrtEngine>(engine_path, 640, 640,
                                            static_cast<float>(conf_thres_));

    camera_intrinsics_ = {get_parameter("fx").as_double(),
                          get_parameter("fy").as_double(),
                          get_parameter("cx").as_double(),
                          get_parameter("cy").as_double()};

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&LineP2pTuningNode::OnImage, this, std::placeholders::_1));
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, rclcpp::SensorDataQoS(),
        std::bind(&LineP2pTuningNode::OnCameraInfo, this,
                  std::placeholders::_1));
    if (use_imu_rectification_ && !assume_zero_imu_) {
      imu_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
          imu_topic_, rclcpp::QoS(10).reliable(),
          std::bind(&LineP2pTuningNode::OnImu, this,
                    std::placeholders::_1));
    }
    const auto command_qos =
        rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    action_pub_ = create_publisher<vision::msg::ActionCommand>(
        action_cmd_topic_, command_qos);
    action_status_sub_ = create_subscription<vision::msg::CommandStatus>(
        action_status_topic_, command_qos,
        std::bind(&LineP2pTuningNode::OnActionStatus, this,
                  std::placeholders::_1));
    start_service_ = create_service<std_srvs::srv::Trigger>(
        "start_line_tuning_trial",
        std::bind(&LineP2pTuningNode::OnStartTrial, this,
                  std::placeholders::_1, std::placeholders::_2));
    parameter_callback_handle_ = add_on_set_parameters_callback(
        std::bind(&LineP2pTuningNode::OnTuningParameters, this,
                  std::placeholders::_1));

    if (inference_hz_ > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / inference_hz_));
      timer_ = create_wall_timer(period, [this]() { ProcessLatestImage(); });
    }
    if (show_debug_view_) {
      cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
    }
    RCLCPP_INFO(get_logger(),
                "LINE P2P tuning node ready. Call /%s/start_line_tuning_trial",
                get_name());
  }

  ~LineP2pTuningNode() override {
    if (show_debug_view_) cv::destroyWindow(kWindowName);
  }

private:
  enum class State { kIdle, kObserving, kDeciding, kWaitingDone, kHolding };
  static constexpr const char *kWindowName = "Line P2P Tuning";

  static const char *StateName(State state) {
    switch (state) {
    case State::kIdle: return "IDLE";
    case State::kObserving: return "OBSERVING";
    case State::kDeciding: return "DECIDING";
    case State::kWaitingDone: return "WAIT_DONE";
    case State::kHolding: return "HOLDING";
    }
    return "UNKNOWN";
  }

  static const char *ActionName(vision_core::MissionAction action) {
    using vision_core::MissionAction;
    switch (action) {
    case MissionAction::kStepForwardOne: return "STEP_FORWARD_ONE";
    case MissionAction::kStepForwardFive: return "STEP_FORWARD_FIVE";
    case MissionAction::kTurnLeft: return "TURN_LEFT";
    case MissionAction::kTurnRight: return "TURN_RIGHT";
    case MissionAction::kStepForwardLeft: return "STEP_FORWARD_LEFT";
    case MissionAction::kStepForwardRight: return "STEP_FORWARD_RIGHT";
    case MissionAction::kTurnLeftAndStep: return "TURN_LEFT_AND_STEP";
    case MissionAction::kTurnRightAndStep: return "TURN_RIGHT_AND_STEP";
    case MissionAction::kContactWalk: return "CONTACT_WALK_RESERVED";
    default: return "NONE";
    }
  }

  std::vector<vision_core::PerceptionDetection> ConvertDetections(
      const std::vector<Detection> &detections) const {
    std::vector<vision_core::PerceptionDetection> output;
    output.reserve(detections.size());
    for (const auto &detection : detections) {
      output.push_back({{{static_cast<double>(detection.box.x),
                          static_cast<double>(detection.box.y),
                          static_cast<double>(detection.box.width),
                          static_cast<double>(detection.box.height)},
                         detection.confidence, detection.class_id},
                        std::nullopt, std::nullopt, std::nullopt});
    }
    return output;
  }

  void OnImage(const sensor_msgs::msg::Image::SharedPtr image) {
    if (inference_hz_ <= 0.0) {
      ProcessImage(image);
      return;
    }
    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_image_ = image;
  }

  void ProcessLatestImage() {
    sensor_msgs::msg::Image::SharedPtr image;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      image = latest_image_;
      latest_image_.reset();
    }
    if (image) ProcessImage(image);
  }

  void OnCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr message) {
    if (message->k[0] <= 0.0 || message->k[4] <= 0.0) return;
    camera_intrinsics_ =
        {message->k[0], message->k[4], message->k[2], message->k[5]};
  }

  void OnImu(const geometry_msgs::msg::Vector3Stamped::SharedPtr message) {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    roll_rad_ = std::clamp(message->vector.x, -imu_abs_limit_rad_,
                           imu_abs_limit_rad_);
    pitch_rad_ = std::clamp(message->vector.y, -imu_abs_limit_rad_,
                            imu_abs_limit_rad_);
    imu_ready_ = true;
  }

  void OnActionStatus(const vision::msg::CommandStatus::SharedPtr message) {
    if (message->command_id == 0) return;
    vision_core::CommandDeliveryFeedback feedback;
    feedback.action_id = message->command_id;
    if (message->status == vision::msg::CommandStatus::ACK) {
      feedback.acknowledged = true;
    } else if (message->status == vision::msg::CommandStatus::READY) {
      feedback.acknowledged = true;
      feedback.ready = true;
    } else if (message->status == vision::msg::CommandStatus::DONE) {
      feedback.acknowledged = true;
      feedback.done = true;
    } else {
      return;
    }
    std::lock_guard<std::mutex> lock(feedback_mutex_);
    feedback_queue_.push_back(feedback);
  }

  void OnStartTrial(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (state_ != State::kIdle) {
      response->success = false;
      response->message = std::string("trial busy: ") + StateName(state_);
      return;
    }
    {
      std::lock_guard<std::mutex> controller_lock(controller_mutex_);
      if (!controller_->UpdateLineP2pTuning(line_p2p_config_)) {
        response->success = false;
        response->message = "previous action is not finished";
        return;
      }
    }
    ++trial_id_;
    if (trial_id_ == 0) ++trial_id_;
    start_sec_ = get_clock()->now().seconds();
    accumulator_.Begin(trial_id_, start_sec_);
    state_ = State::kObserving;
    action_id_ = 0;
    last_action_ = vision_core::MissionAction::kNone;
    response->success = true;
    response->message = "collecting for " + std::to_string(observation_sec_) +
                        " sec";
    RCLCPP_INFO(get_logger(), "Trial %lu: collecting for %.2f sec",
                static_cast<unsigned long>(trial_id_), observation_sec_);
  }

  rcl_interfaces::msg::SetParametersResult OnTuningParameters(
      const std::vector<rclcpp::Parameter> &parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    auto next_line = line_p2p_config_;
    double next_observation = observation_sec_;
    bool touched = false;
    for (const auto &parameter : parameters) {
      double *target = nullptr;
      const auto &name = parameter.get_name();
      if (name == "line_p2p_offset_gain") target = &next_line.offset_gain;
      else if (name == "line_p2p_heading_gain") target = &next_line.heading_gain;
      else if (name == "line_p2p_steering_deadband") target = &next_line.steering_deadband;
      else if (name == "line_tuning_observation_sec") target = &next_observation;
      if (target == nullptr) continue;
      touched = true;
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) {
        result.successful = false;
        result.reason = name + " must be double";
        return result;
      }
      *target = parameter.as_double();
    }
    if (!touched) return result;
    if (state_ != State::kIdle) {
      result.successful = false;
      result.reason = "parameters are locked until action DONE";
      return result;
    }
    const bool valid =
        std::isfinite(next_line.offset_gain) && next_line.offset_gain >= 0.0 &&
        std::isfinite(next_line.heading_gain) && next_line.heading_gain >= 0.0 &&
        std::isfinite(next_line.steering_deadband) &&
        next_line.steering_deadband >= 0.0 &&
        std::isfinite(next_observation) && next_observation > 0.0;
    if (!valid) {
      result.successful = false;
      result.reason = "invalid gain/threshold ordering or value";
      return result;
    }
    {
      std::lock_guard<std::mutex> controller_lock(controller_mutex_);
      if (!controller_->UpdateLineP2pTuning(next_line)) {
        result.successful = false;
        result.reason = "core still has an active action";
        return result;
      }
    }
    line_p2p_config_ = next_line;
    observation_sec_ = next_observation;
    return result;
  }

  void PrepareInput(double now_sec,
                    vision_core::PerceptionFrameInput &input) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    input.allow_new_line_action = false;
    if (state_ == State::kHolding) {
      if (now_sec + 1e-9 >= hold_until_sec_) {
        state_ = State::kIdle;
        hold_until_sec_ = 0.0;
        RCLCPP_INFO(get_logger(), "Pose hold finished; trigger unlocked");
      }
      return;
    }
    if (state_ != State::kObserving ||
        now_sec - start_sec_ < observation_sec_) {
      return;
    }
    const auto guide = accumulator_.FinishAll(trial_id_);
    if (!guide) {
      state_ = State::kHolding;
      hold_until_sec_ = now_sec + hold_sec_;
      RCLCPP_WARN(get_logger(),
                  "No valid LineGuide; holding current pose for %.2f sec",
                  hold_sec_);
      return;
    }
    last_guide_ = *guide;
    input.line_decision_guide_override = *guide;
    input.allow_new_line_action = true;
    state_ = State::kDeciding;
  }

  void FinishStep(
      double now_sec,
      const vision_core::CommandDeliveryFeedback &delivered_feedback,
      const vision_core::PerceptionMissionFrameResult &result) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (state_ == State::kObserving) {
      accumulator_.Add(trial_id_, now_sec,
                       result.mission.line_features.guide);
      return;
    }
    if (state_ == State::kDeciding) {
      if (result.mission.command.command_type ==
              vision_core::CommandType::kAction &&
          result.mission.command.action_id != 0) {
        action_id_ = result.mission.command.action_id;
        last_action_ = result.mission.command.action;
        state_ = State::kWaitingDone;
        RCLCPP_INFO(get_logger(),
                    "Trial action: id=%lu action=%u(%s); waiting for executor "
                    "ACK/DONE",
                    static_cast<unsigned long>(action_id_),
                    static_cast<unsigned>(last_action_),
                    ActionName(last_action_));
      } else {
        state_ = State::kHolding;
        hold_until_sec_ = now_sec + hold_sec_;
        RCLCPP_INFO(get_logger(),
                    "No line action selected; holding current pose for %.2f sec",
                    hold_sec_);
      }
      return;
    }
    if (state_ == State::kWaitingDone && delivered_feedback.done &&
        delivered_feedback.action_id == action_id_ &&
        result.mission.command.action_id == 0) {
      state_ = State::kIdle;
      action_id_ = 0;
      RCLCPP_INFO(get_logger(), "Action DONE: parameters unlocked");
    }
  }

  void PublishAction(const vision_core::ControlCommand &command) {
    if (command.command_type != vision_core::CommandType::kAction) return;
    vision::msg::ActionCommand message;
    message.action_id = command.action_id;
    message.mission =
        static_cast<std::uint8_t>(vision_core::MissionType::kLine);
    message.action = static_cast<std::uint16_t>(command.action);
    message.target_yaw_deg = command.target_yaw_deg;
    action_pub_->publish(message);
  }

  void ProcessImage(const sensor_msgs::msg::Image::SharedPtr &message) {
    cv_bridge::CvImageConstPtr image;
    try {
      image = cv_bridge::toCvShare(message, "bgr8");
    } catch (const cv_bridge::Exception &error) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", error.what());
      return;
    }
    std::vector<Detection> detections;
    if (!yolo_->Infer(image->image, detections)) return;

    vision_core::CommandDeliveryFeedback feedback;
    {
      std::lock_guard<std::mutex> lock(feedback_mutex_);
      if (!feedback_queue_.empty()) {
        feedback = feedback_queue_.front();
        feedback_queue_.pop_front();
      }
    }
    double roll = 0.0;
    double pitch = 0.0;
    bool imu_valid = false;
    if (use_imu_rectification_ && assume_zero_imu_) {
      imu_valid = true;
    } else if (use_imu_rectification_) {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      roll = roll_rad_;
      pitch = pitch_rad_;
      imu_valid = imu_ready_;
    }

    const double now_sec = get_clock()->now().seconds();
    vision_core::PerceptionFrameInput input;
    input.detections = ConvertDetections(detections);
    input.intrinsics = camera_intrinsics_;
    input.enable_imu_rectification = use_imu_rectification_;
    input.imu_valid = imu_valid;
    input.roll_rad = roll;
    input.pitch_rad = pitch;
    input.image_width = image->image.cols;
    input.image_height = image->image.rows;
    input.now_sec = now_sec;
    input.command_transport_enabled = true;
    input.delivery_feedback = feedback;
    PrepareInput(now_sec, input);

    vision_core::PerceptionMissionFrameResult result;
    {
      std::lock_guard<std::mutex> lock(controller_mutex_);
      result = controller_->StepPerception(input);
    }
    FinishStep(now_sec, feedback, result);
    PublishAction(result.mission.command);

    if (show_debug_view_) DrawDebug(image->image, detections, result);
  }

  void DrawDebug(const cv::Mat &image,
                 const std::vector<Detection> &detections,
                 const vision_core::PerceptionMissionFrameResult &result) {
    cv::Mat view = image.clone();
    for (const auto &detection : detections) {
      if (detection.class_id != line_class_id_ ||
          detection.confidence < conf_thres_) {
        continue;
      }
      cv::rectangle(view, detection.box, cv::Scalar(0, 255, 0), 2);
    }
    State state;
    vision_core::LineGuide guide = result.mission.line_features.guide;
    vision_core::LineP2pConfig gains;
    vision_core::MissionAction action;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      state = state_;
      gains = line_p2p_config_;
      action = last_action_;
      if ((state == State::kDeciding || state == State::kWaitingDone) &&
          last_guide_.valid) {
        guide = last_guide_;
      }
    }
    char text[320];
    int y = 26;
    const auto draw = [&](const cv::Scalar &color) {
      cv::putText(view, text, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                  0.55, color, 2);
      y += 23;
    };
    std::snprintf(text, sizeof(text), "STATE: %s", StateName(state));
    draw(cv::Scalar(0, 255, 255));
    std::snprintf(text, sizeof(text), "O=%+.3f H=%+.3f C=%+.3f(%s)",
                  guide.offset, guide.heading_rad, guide.curvature_rad,
                  guide.curvature_valid ? "valid" : "invalid");
    draw(cv::Scalar(255, 255, 0));
    const double offset_term = gains.offset_gain * guide.offset;
    const double heading_term = gains.heading_gain * guide.heading_rad;
    const double score = offset_term + heading_term;
    std::snprintf(text, sizeof(text), "TERMS O=%+.3f H=%+.3f",
                  offset_term, heading_term);
    draw(cv::Scalar(255, 255, 0));
    std::snprintf(text, sizeof(text), "SCORE=%+.3f BAND=+/-%.3f",
                  score, gains.steering_deadband);
    draw(cv::Scalar(255, 255, 0));
    std::snprintf(text, sizeof(text), "ACTION: %u %s",
                  static_cast<unsigned>(action), ActionName(action));
    draw(cv::Scalar(0, 200, 255));
    cv::imshow(kWindowName, view);
    cv::waitKey(1);
  }

  std::unique_ptr<YoloTrtEngine> yolo_;
  std::unique_ptr<vision_core::MissionController> controller_;
  std::mutex controller_mutex_;
  vision_core::LineP2pConfig line_p2p_config_{};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr imu_sub_;
  rclcpp::Subscription<vision::msg::CommandStatus>::SharedPtr
      action_status_sub_;
  rclcpp::Publisher<vision::msg::ActionCommand>::SharedPtr action_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_service_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameter_callback_handle_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex image_mutex_;
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  std::mutex imu_mutex_;
  bool imu_ready_{false};
  double roll_rad_{0.0};
  double pitch_rad_{0.0};
  double imu_abs_limit_rad_{M_PI / 4.0};
  vision_core::Intrinsics camera_intrinsics_{};

  std::mutex feedback_mutex_;
  std::deque<vision_core::CommandDeliveryFeedback> feedback_queue_;
  std::mutex state_mutex_;
  State state_{State::kIdle};
  std::uint64_t trial_id_{0};
  std::uint64_t action_id_{0};
  double start_sec_{0.0};
  double observation_sec_{1.5};
  double hold_sec_{0.0};
  double hold_until_sec_{0.0};
  vision_core::LineGuideAccumulator accumulator_;
  vision_core::LineGuide last_guide_{};
  vision_core::MissionAction last_action_{vision_core::MissionAction::kNone};

  int line_class_id_{0};
  double conf_thres_{0.6};
  bool use_imu_rectification_{false};
  bool assume_zero_imu_{false};
  double inference_hz_{15.0};
  bool show_debug_view_{true};
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string imu_topic_;
  std::string action_cmd_topic_;
  std::string action_status_topic_;
};

} // namespace vision

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vision::LineP2pTuningNode>());
  rclcpp::shutdown();
  return 0;
}
