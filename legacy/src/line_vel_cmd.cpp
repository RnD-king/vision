// 사용 중단: vision_core로 이식되기 전의 ROS Twist 어댑터 구현.
//
// line feature에서 rule 기반 속도 명령을 계산한다.

#include "vision/line_vel_cmd.hpp"

#include <algorithm>
#include <cmath>

namespace vision {

namespace {
vision_core::RuleConfig ToCoreConfig(const LineVelCmd::Config &cfg) {
  vision_core::RuleConfig out;
  out.cmd_vx_min = cfg.cmd_vx_min;
  out.cmd_vx_max = cfg.cmd_vx_max;
  out.cmd_wz_min = cfg.cmd_wz_min;
  out.cmd_wz_max = cfg.cmd_wz_max;
  out.v_base = cfg.v_base;
  out.k_u = cfg.k_u;
  out.k_slope = cfg.k_slope;
  out.k_v_u = cfg.k_v_u;
  out.k_v_slope = cfg.k_v_slope;
  out.dv_max = cfg.dv_max;
  out.dw_max = cfg.dw_max;
  out.recover_vx = cfg.recover_vx;
  out.recover_wz = cfg.recover_wz;
  out.low_visible_n = cfg.low_visible_n;
  out.no_visible_n = cfg.no_visible_n;
  out.low_visible_vx = cfg.low_visible_vx;
  out.no_visible_vx = cfg.no_visible_vx;
  out.low_visible_wz_decay = cfg.low_visible_wz_decay;
  out.no_visible_wz_decay = cfg.no_visible_wz_decay;
  return out;
}
} // namespace

LineVelCmd::LineVelCmd(const Config &cfg)
    : cfg_(cfg), controller_(ToCoreConfig(cfg)) {}

LineVelCmd::Result LineVelCmd::Compute(const LineFeatureExtractor::Features &features) {
  const vision_core::Features core_features{
      features.u_err_near, features.u_err_lookahead, features.u_err_ctrl,
      features.slope, features.n_visible, features.in_recovery,
      features.vx_prev, features.wz_prev};
  controller_.Observe(core_features, {});
  const auto command = controller_.Compute(core_features);

  Result result;
  result.twist.linear.x = command.vx;
  result.twist.angular.z = command.wz;
  result.u_err_ctrl = features.u_err_ctrl;
  result.slope = features.slope;
  result.n_visible = features.n_visible;
  result.in_recovery = features.in_recovery > 0.5;
  return result;
}

} // namespace vision
