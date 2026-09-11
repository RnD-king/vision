#pragma once

// 사용 중단: 현재 line_perception_node는
// vision_core/line_velocity_controller.hpp를 직접 호출한다.

// Pipeline command stage for line following.
// line feature에서 rule 기반 속도 명령을 계산한다. ROS publish는 node에서만 한다.

#include <geometry_msgs/msg/twist.hpp>

#include "vision/line_feature_extractor.hpp"
#include "vision_core/line_velocity_controller.hpp"

namespace vision {

class LineVelCmd {
public:
  struct Config {
    double cmd_vx_min{0.10};
    double cmd_vx_max{1.20};
    double cmd_wz_min{-1.90};
    double cmd_wz_max{1.90};
    double v_base{0.85};
    double k_u{3.00};
    double k_slope{3.00};
    double k_v_u{0.35};
    double k_v_slope{0.25};
    double dv_max{0.12};
    double dw_max{0.40};
    double recover_vx{0.12};
    double recover_wz{0.75};
    double low_visible_n{2.0};
    double no_visible_n{0.5};
    double low_visible_vx{0.18};
    double no_visible_vx{0.10};
    double low_visible_wz_decay{0.90};
    double no_visible_wz_decay{0.95};
  };

  struct Result {
    geometry_msgs::msg::Twist twist;
    double u_err_ctrl{0.0};
    double slope{0.0};
    double n_visible{0.0};
    bool in_recovery{false};
  };

  explicit LineVelCmd(const Config &cfg);

  Result Compute(const LineFeatureExtractor::Features &features);

private:
  Config cfg_;
  vision_core::LineVelocityController controller_;
};

} // namespace vision
