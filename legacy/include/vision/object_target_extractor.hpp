#pragma once

// 사용 중단: 객체 target 선택은 vision_core/object_target_extractor.hpp가 담당한다.

// Pipeline step 5-b interface.
// 현재는 ball/goal/backboard/hurdle detection을 단일 object target으로 고르는
// skeleton이다. bbox 크기 정보는 원본 그대로 유지하고, 중심점만 이후 IMU 보정
// 대상이 된다.

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "vision/types.hpp"

namespace vision {

class ObjectTargetExtractor {
public:
  struct Config {
    int ball_class_id{1};
    int goal_class_id{2};
    int backboard_class_id{3};
    int hurdle_class_id{4};
    float ball_conf_thres{0.60f};
    float goal_conf_thres{0.60f};
    float backboard_conf_thres{0.60f};
    float hurdle_conf_thres{0.60f};
    int min_box_width{2};
    int min_box_height{2};
  };

  struct Target {
    int class_id{0};
    float confidence{0.0f};
    cv::Rect box_px;
    cv::Point2f center_px;
    cv::Point2f rectified_center_px;
    float width_px{0.0f};
    float height_px{0.0f};
    float area_px{0.0f};
    bool center_rectified{false};
  };

  explicit ObjectTargetExtractor(const Config &cfg);

  // 현재는 ball/goal/backboard/hurdle를 단일 target 후보로 뽑는 skeleton이다.
  // 반환값은 원본 이미지 기준 bbox/center이며, IMU 보정은 node에서 center에만
  // 적용한다.
  std::optional<Target> ExtractBall(const std::vector<Detection> &detections) const;
  std::optional<Target> ExtractGoal(const std::vector<Detection> &detections) const;
  std::optional<Target> ExtractBackboard(const std::vector<Detection> &detections) const;
  std::optional<Target> ExtractHurdle(const std::vector<Detection> &detections) const;

private:
  std::optional<Target> ExtractBestByClass(const std::vector<Detection> &detections, int class_id,
                                           float conf_thres) const;

  bool IsValidBox(const cv::Rect &box) const;
  static Target MakeTarget(const Detection &det);

  Config cfg_;
};

} // namespace vision
