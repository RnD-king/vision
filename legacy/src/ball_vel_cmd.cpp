// 사용 중단: vision_core로 이식되기 전의 ROS Twist 기반 공 제어 구현.
//
// ball target 기반 접근 속도 명령을 계산한다. 45도 카메라에서는 공의 화면
// 가로 위치로 부드러운 heading 제어만 하고, 공이 화면 아래쪽으로 충분히
// 내려오면 카메라 하향 전환 구간으로 넘어간다.

#include "vision/ball_vel_cmd.hpp"

#include <algorithm>
#include <cmath>

namespace vision {

namespace {

double SafeDenom(double v) {
  return std::max(v, 1e-6);
}

} // namespace

BallVelCmd::BallVelCmd(const Config &cfg) : cfg_(cfg) {}

const char *BallVelCmd::ModeName(Mode mode) {
  switch (mode) {
  case Mode::kLineFollow:
    return "LINE_FOLLOW";
  case Mode::kApproachFar:
    return "BALL_APPROACH_FAR";
  case Mode::kTiltDownWalk:
    return "CAMERA_TILT_DOWN_WALK";
  case Mode::kApproachNear:
    return "BALL_APPROACH_NEAR";
  case Mode::kStopForPickup:
    return "STOP_FOR_PICKUP";
  }
  return "UNKNOWN";
}

double BallVelCmd::Clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

double BallVelCmd::LimitRate(double prev, double target, double delta) {
  return Clamp(target, prev - delta, prev + delta);
}

BallVelCmd::Result BallVelCmd::Compute(const std::optional<ObjectTargetExtractor::Target> &ball_target,
                                       const cv::Size &image_size, double now_sec) {
  UpdateTracker(ball_target, image_size);

  if (mode_ == Mode::kLineFollow && smoothed_.stable) {
    mode_ = Mode::kApproachFar;
    state_enter_sec_ = now_sec;
  }

  if ((mode_ == Mode::kApproachFar || mode_ == Mode::kApproachNear) && lost_count_ > cfg_.lost_frames) {
    ResetToLineFollow();
  }

  Result result;
  result.mode = mode_;
  result.tracked = smoothed_;

  switch (mode_) {
  case Mode::kLineFollow:
    result.active = false;
    return result;

  case Mode::kApproachFar: {
    result.active = true;
    if (smoothed_.visible) {
      result.twist = ComputeFarCmd(smoothed_);
      PushRecentCmd(result.twist);
      const bool should_tilt = smoothed_.v_norm >= cfg_.tilt_down_v_norm || smoothed_.h_norm >= cfg_.tilt_down_h_norm;
      if (should_tilt) {
        mode_ = Mode::kTiltDownWalk;
        state_enter_sec_ = now_sec;
        result.mode = mode_;
        result.request_camera_down = true;
      }
    } else {
      result.twist = last_cmd_;
      result.twist.linear.x *= 0.5;
      result.twist.angular.z *= 0.5;
    }
    last_cmd_ = result.twist;
    return result;
  }

  case Mode::kTiltDownWalk: {
    result.active = true;
    result.request_camera_down = true;
    result.twist = ComputeHoldCmd();
    last_cmd_ = result.twist;

    const double elapsed = now_sec - state_enter_sec_;
    if (elapsed >= cfg_.camera_tilt_duration_sec + cfg_.camera_settle_sec) {
      mode_ = Mode::kApproachNear;
      state_enter_sec_ = now_sec;
      result.mode = mode_;
    }
    return result;
  }

  case Mode::kApproachNear: {
    result.active = true;
    bool reached = false;
    if (smoothed_.visible) {
      result.twist = ComputeNearCmd(smoothed_, &reached);
    } else {
      result.twist = geometry_msgs::msg::Twist{};
    }
    last_cmd_ = result.twist;
    if (reached) {
      mode_ = Mode::kStopForPickup;
      state_enter_sec_ = now_sec;
      result.mode = mode_;
      result.reached_pickup_pose = true;
      result.twist = geometry_msgs::msg::Twist{};
      last_cmd_ = result.twist;
    }
    return result;
  }

  case Mode::kStopForPickup:
    result.active = true;
    result.reached_pickup_pose = true;
    result.twist = geometry_msgs::msg::Twist{};
    return result;
  }

  return result;
}

void BallVelCmd::UpdateTracker(const std::optional<ObjectTargetExtractor::Target> &ball_target,
                               const cv::Size &image_size) {
  const bool image_valid = image_size.width > 1 && image_size.height > 1;
  const bool detected = ball_target.has_value() && image_valid;

  hit_history_.push_back(detected);
  while (static_cast<int>(hit_history_.size()) > std::max(1, cfg_.stable_window)) {
    hit_history_.pop_front();
  }

  if (detected) {
    lost_count_ = 0;
    TrackedBall obs;
    obs.visible = true;
    obs.center_px = ball_target->center_rectified ? ball_target->rectified_center_px : ball_target->center_px;
    obs.u_norm = Clamp(static_cast<double>(obs.center_px.x) / SafeDenom(static_cast<double>(image_size.width)), 0.0, 1.0);
    obs.v_norm = Clamp(static_cast<double>(obs.center_px.y) / SafeDenom(static_cast<double>(image_size.height)), 0.0, 1.0);
    obs.h_norm = Clamp(static_cast<double>(ball_target->height_px) / SafeDenom(static_cast<double>(image_size.height)), 0.0, 1.0);
    obs.area_norm =
        Clamp(static_cast<double>(ball_target->area_px) /
                  SafeDenom(static_cast<double>(image_size.width) * static_cast<double>(image_size.height)),
              0.0, 1.0);
    obs.confidence = ball_target->confidence;

    const double alpha = Clamp(cfg_.smooth_alpha, 0.0, 1.0);
    if (!has_smoothed_) {
      smoothed_ = obs;
      has_smoothed_ = true;
    } else {
      smoothed_.visible = true;
      smoothed_.center_px.x = static_cast<float>((1.0 - alpha) * smoothed_.center_px.x + alpha * obs.center_px.x);
      smoothed_.center_px.y = static_cast<float>((1.0 - alpha) * smoothed_.center_px.y + alpha * obs.center_px.y);
      smoothed_.u_norm = (1.0 - alpha) * smoothed_.u_norm + alpha * obs.u_norm;
      smoothed_.v_norm = (1.0 - alpha) * smoothed_.v_norm + alpha * obs.v_norm;
      smoothed_.h_norm = (1.0 - alpha) * smoothed_.h_norm + alpha * obs.h_norm;
      smoothed_.area_norm = (1.0 - alpha) * smoothed_.area_norm + alpha * obs.area_norm;
      smoothed_.confidence = obs.confidence;
    }
  } else {
    ++lost_count_;
    smoothed_.visible = false;
  }

  const int hits = static_cast<int>(std::count(hit_history_.begin(), hit_history_.end(), true));
  smoothed_.stable = hits >= std::max(1, cfg_.stable_min_hits);
}

geometry_msgs::msg::Twist BallVelCmd::ComputeFarCmd(const TrackedBall &ball) {
  const double u_err = (ball.u_norm - cfg_.far_u_des_norm) / 0.5;
  const double wz_raw = -cfg_.far_wz_max * std::tanh(cfg_.far_heading_gain * u_err);
  const double turn_ratio = Clamp(std::abs(wz_raw) / SafeDenom(cfg_.far_wz_max), 0.0, 1.0);
  const double slow_by_turn = Clamp(cfg_.far_slow_by_turn, 0.0, 1.0);
  const double vx_raw = std::max(cfg_.far_vx_min, cfg_.far_vx * (1.0 - slow_by_turn * turn_ratio));

  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = LimitRate(last_cmd_.linear.x, vx_raw, cfg_.far_dv_max);
  cmd.angular.z = LimitRate(last_cmd_.angular.z, wz_raw, cfg_.far_dw_max);
  return cmd;
}

geometry_msgs::msg::Twist BallVelCmd::ComputeHoldCmd() const {
  geometry_msgs::msg::Twist cmd;
  if (recent_cmds_.empty()) {
    cmd.linear.x = cfg_.hold_default_vx;
    return cmd;
  }

  for (const auto &recent : recent_cmds_) {
    cmd.linear.x += recent.linear.x;
    cmd.angular.z += recent.angular.z;
  }
  const double inv_n = 1.0 / static_cast<double>(recent_cmds_.size());
  cmd.linear.x *= inv_n;
  cmd.angular.z *= inv_n;

  cmd.linear.x = Clamp(cmd.linear.x, cfg_.hold_vx_min, cfg_.hold_vx_max);
  cmd.angular.z = Clamp(cmd.angular.z, -cfg_.hold_wz_max, cfg_.hold_wz_max);
  return cmd;
}

geometry_msgs::msg::Twist BallVelCmd::ComputeNearCmd(const TrackedBall &ball, bool *reached) const {
  const double x_err = ball.u_norm - cfg_.near_target_u_norm;
  const double y_err = ball.v_norm - cfg_.near_target_v_norm;

  if (reached) {
    *reached = std::abs(x_err) <= cfg_.near_x_tol && std::abs(y_err) <= cfg_.near_y_tol;
  }

  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = Clamp(-cfg_.near_ky * y_err, -cfg_.near_vx_max, cfg_.near_vx_max);
  if (cfg_.near_use_lateral) {
    cmd.linear.y = Clamp(-cfg_.near_kx * x_err, -cfg_.near_vy_max, cfg_.near_vy_max);
    cmd.angular.z = 0.0;
  } else {
    cmd.linear.y = 0.0;
    cmd.angular.z = Clamp(-cfg_.near_wz_gain * x_err, -cfg_.near_wz_max, cfg_.near_wz_max);
  }
  return cmd;
}

void BallVelCmd::PushRecentCmd(const geometry_msgs::msg::Twist &cmd) {
  recent_cmds_.push_back(cmd);
  while (static_cast<int>(recent_cmds_.size()) > std::max(1, cfg_.hold_cmd_window)) {
    recent_cmds_.pop_front();
  }
}

void BallVelCmd::ResetToLineFollow() {
  mode_ = Mode::kLineFollow;
  lost_count_ = 0;
  hit_history_.clear();
  has_smoothed_ = false;
  smoothed_ = TrackedBall{};
  recent_cmds_.clear();
  last_cmd_ = geometry_msgs::msg::Twist{};
}

} // namespace vision
