#pragma once

// 사용 중단: detection 중심점 추출은 LineDetectionAdapter가, 특징 계산은
// vision_core/line_feature_extractor.hpp가 담당한다.

// Pipeline step 5-a/7 interface.
// line class detection에서 중심점을 추출하고, 보정된 line 중심점 배열을
// rule/policy 호환 8D feature로 변환한다.

#include <opencv2/core.hpp>
#include <vector>

#include "vision/types.hpp"

namespace vision {

class LineFeatureExtractor {
public:
  struct Config {
    int line_class_id{0};
    float conf_thres{0.60f};
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
    // 학습된 g1_vision policy 입력과 동일한 8D 순서.
    double u_err_near{0.0};
    double u_err_lookahead{0.0};
    double u_err_ctrl{0.0};
    double slope{0.0};
    double n_visible{0.0};
    double in_recovery{0.0};
    double vx_prev{0.0};
    double wz_prev{0.0};
  };

  explicit LineFeatureExtractor(const Config &cfg);

  // YOLO Detection들 중 line class만 골라 bbox 중심점을 반환한다.
  // 반환 좌표는 원본 이미지 픽셀 좌표이며 y 내림차순으로 정렬된다.
  std::vector<cv::Point2f> ExtractCenters(const std::vector<Detection> &dets) const;

  // 보정된 픽셀 점들을 8D feature로 변환한다.
  Features ComputeFeatures(const std::vector<cv::Point2f> &pts_px, const cv::Size &image_size,
                           bool previous_in_recovery, double vx_prev, double wz_prev) const;

private:
  static constexpr double kEps = 1e-9;

  Config cfg_;
};

} // namespace vision
