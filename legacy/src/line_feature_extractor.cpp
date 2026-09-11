// 사용 중단: 중심점 추출과 특징 계산이 분리되기 전의 어댑터 구현.
//
// Pipeline step 5-a/7.
// YOLO detection에서 line class bbox 중심점을 추출하고, IMU 보정 이후의
// 중심점들을 rule/policy 호환 8D feature로 변환한다.

#include "vision/line_feature_extractor.hpp"
#include "vision_core/line_feature_extractor.hpp"

#include <algorithm>
#include <cmath>

namespace vision {

namespace {

double Clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

} // namespace

LineFeatureExtractor::LineFeatureExtractor(const Config &cfg) : cfg_(cfg) {}

std::vector<cv::Point2f> LineFeatureExtractor::ExtractCenters(const std::vector<Detection> &dets) const {
  std::vector<cv::Point2f> centers;
  centers.reserve(dets.size());

  for (const auto &d : dets) {
    if (d.class_id != cfg_.line_class_id) {
      continue;
    }
    if (d.confidence < cfg_.conf_thres) {
      continue;
    }

    const auto &b = d.box;
    if (b.width <= 1 || b.height <= 1) {
      continue;
    }

    const float cx = static_cast<float>(b.x) + 0.5f * static_cast<float>(b.width);
    const float cy = static_cast<float>(b.y) + 0.5f * static_cast<float>(b.height);
    centers.emplace_back(cx, cy);
  }

  std::sort(centers.begin(), centers.end(), [](const cv::Point2f &a, const cv::Point2f &b) {
    if (a.y == b.y) {
      return a.x < b.x;
    }
    return a.y > b.y;
  });

  return centers;
}

LineFeatureExtractor::Features LineFeatureExtractor::ComputeFeatures(const std::vector<cv::Point2f> &pts_px,
                                                                     const cv::Size &image_size,
                                                                     bool previous_in_recovery, double vx_prev,
                                                                     double wz_prev) const {
  std::vector<vision_core::Point2> points;
  points.reserve(pts_px.size());
  for (const auto &point : pts_px) points.push_back({point.x, point.y});
  vision_core::FeatureConfig cfg;
  cfg.max_centers = cfg_.max_centers;
  cfg.image_center_u = cfg_.image_center_u;
  cfg.lookahead_delta_v_px = cfg_.lookahead_delta_v_px;
  cfg.lookahead_alpha_normal = cfg_.lookahead_alpha_normal;
  cfg.lookahead_alpha_recovery = cfg_.lookahead_alpha_recovery;
  cfg.recover_enter_nvis = cfg_.recover_enter_nvis;
  cfg.recover_exit_nvis = cfg_.recover_exit_nvis;
  cfg.recover_enter_u = cfg_.recover_enter_u;
  cfg.recover_exit_u = cfg_.recover_exit_u;
  const auto core = vision_core::ComputeLineFeatures(
      points, image_size.width, image_size.height, previous_in_recovery,
      vx_prev, wz_prev, cfg);
  return {core.u_err_near, core.u_err_lookahead, core.u_err_ctrl, core.slope,
          core.n_visible, core.in_recovery, core.vx_prev, core.wz_prev};
}

} // namespace vision
