#pragma once

// 사용 중단: 현재 특징 계산은 /home/noh/vision_core의
// line_feature_extractor를 사용한다.

#include <opencv2/core.hpp>
#include <vector>

namespace vision {

class FeatureExtractor {
public:
  struct Config {
    int max_centers{8};
    double image_center_u{-1.0};
    double lookahead_delta_v_px{190.0};
    double lookahead_alpha_normal{0.70};
    double lookahead_alpha_recovery{0.20};
    double recover_enter_nvis{2.0};
    double recover_exit_nvis{3.0};
    double recover_enter_u{0.70};
    double recover_exit_u{0.35};
  };

  struct Features {
    double u_err_near{0.0};
    double u_err_lookahead{0.0};
    double u_err_ctrl{0.0};
    double slope{0.0};
    double n_visible{0.0};
    double in_recovery{0.0};
    double vx_prev{0.0};
    double wz_prev{0.0};
  };

  FeatureExtractor() = default;
  Features Compute(const std::vector<cv::Point2f> &pts_px,
                   const cv::Size &image_size, bool previous_in_recovery,
                   double vx_prev, double wz_prev, const Config &cfg) const;
};

} // namespace vision
