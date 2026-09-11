#pragma once

// 사용 중단: 공 추적과 속도 계산은 vision_core/ball_controller.hpp가 담당한다.

// Pipeline command stage for ball approach.
// ball target 안정화, 45도 카메라 coarse approach, 카메라 하향 전환 중
// 직전 명령 유지, 하향 카메라 near alignment skeleton을 한 파일에 둔다.

#include <deque>
#include <optional>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <opencv2/core.hpp>

#include "vision/object_target_extractor.hpp"

namespace vision {

class BallVelCmd {
public:
  enum class Mode {
    kLineFollow = 0,
    kApproachFar,
    kTiltDownWalk,
    kApproachNear,
    kStopForPickup,
  };

  struct Config {
    int stable_window{5};
    int stable_min_hits{3};
    int lost_frames{5};
    double smooth_alpha{0.45};

    double far_u_des_norm{0.50};
    double far_vx{0.35};
    double far_vx_min{0.10};
    double far_wz_max{0.80};
    double far_heading_gain{2.50};
    double far_slow_by_turn{0.60};
    double far_dv_max{0.12};
    double far_dw_max{0.35};

    double tilt_down_v_norm{0.75};
    double tilt_down_h_norm{0.18};
    double camera_tilt_duration_sec{0.50};
    double camera_settle_sec{0.25};

    int hold_cmd_window{5};
    double hold_vx_min{0.15};
    double hold_vx_max{0.25};
    double hold_wz_max{0.25};
    double hold_default_vx{0.18};

    double near_target_u_norm{0.50};
    double near_target_v_norm{0.70};
    double near_kx{0.50};
    double near_ky{0.45};
    double near_wz_gain{0.80};
    double near_vx_max{0.18};
    double near_vy_max{0.15};
    double near_wz_max{0.50};
    double near_x_tol{0.06};
    double near_y_tol{0.06};
    bool near_use_lateral{true};
  };

  struct TrackedBall {
    bool stable{false};
    bool visible{false};
    cv::Point2f center_px;
    double u_norm{0.0};
    double v_norm{0.0};
    double h_norm{0.0};
    double area_norm{0.0};
    float confidence{0.0f};
  };

  struct Result {
    bool active{false};
    bool reached_pickup_pose{false};
    bool request_camera_down{false};
    Mode mode{Mode::kLineFollow};
    TrackedBall tracked;
    geometry_msgs::msg::Twist twist;
  };

  explicit BallVelCmd(const Config &cfg);

  Result Compute(const std::optional<ObjectTargetExtractor::Target> &ball_target, const cv::Size &image_size,
                 double now_sec);

  static const char *ModeName(Mode mode);

private:
  static double Clamp(double v, double lo, double hi);
  static double LimitRate(double prev, double target, double delta);

  void UpdateTracker(const std::optional<ObjectTargetExtractor::Target> &ball_target, const cv::Size &image_size);
  geometry_msgs::msg::Twist ComputeFarCmd(const TrackedBall &ball);
  geometry_msgs::msg::Twist ComputeHoldCmd() const;
  geometry_msgs::msg::Twist ComputeNearCmd(const TrackedBall &ball, bool *reached) const;
  void PushRecentCmd(const geometry_msgs::msg::Twist &cmd);
  void ResetToLineFollow();

  Config cfg_;
  Mode mode_{Mode::kLineFollow};
  std::deque<bool> hit_history_;
  int lost_count_{0};
  bool has_smoothed_{false};
  TrackedBall smoothed_;
  std::deque<geometry_msgs::msg::Twist> recent_cmds_;
  geometry_msgs::msg::Twist last_cmd_;
  double state_enter_sec_{0.0};
};

} // namespace vision
