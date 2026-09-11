// 사용 중단: vision_core로 이식되기 전의 OpenCV 객체 target 구현.
//
// Pipeline step 5-b.
// YOLO detection 중 ball/goal/backboard/hurdle class를 단일 목표 target으로
// 추출한다.
// 라인의 여러 중심점과 달리 object target은 bbox, center, confidence, size를
// 함께 유지한다. IMU 보정은 다음 단계에서 center에만 적용하고 bbox size는 원본
// 값을 그대로 쓴다.

#include "vision/object_target_extractor.hpp"

#include <algorithm>

namespace vision {

ObjectTargetExtractor::ObjectTargetExtractor(const Config &cfg) : cfg_(cfg) {}

std::optional<ObjectTargetExtractor::Target>
ObjectTargetExtractor::ExtractBall(const std::vector<Detection> &detections) const {
  return ExtractBestByClass(detections, cfg_.ball_class_id, cfg_.ball_conf_thres);
}

std::optional<ObjectTargetExtractor::Target>
ObjectTargetExtractor::ExtractGoal(const std::vector<Detection> &detections) const {
  return ExtractBestByClass(detections, cfg_.goal_class_id, cfg_.goal_conf_thres);
}

std::optional<ObjectTargetExtractor::Target>
ObjectTargetExtractor::ExtractBackboard(const std::vector<Detection> &detections) const {
  return ExtractBestByClass(detections, cfg_.backboard_class_id, cfg_.backboard_conf_thres);
}

std::optional<ObjectTargetExtractor::Target>
ObjectTargetExtractor::ExtractHurdle(const std::vector<Detection> &detections) const {
  return ExtractBestByClass(detections, cfg_.hurdle_class_id, cfg_.hurdle_conf_thres);
}

std::optional<ObjectTargetExtractor::Target>
ObjectTargetExtractor::ExtractBestByClass(const std::vector<Detection> &detections, int class_id,
                                          float conf_thres) const {
  std::optional<Target> best;

  for (const auto &det : detections) {
    if (det.class_id != class_id) {
      continue;
    }
    if (det.confidence < conf_thres) {
      continue;
    }
    if (!IsValidBox(det.box)) {
      continue;
    }

    Target target = MakeTarget(det);
    if (!best) {
      best = target;
      continue;
    }

    const bool higher_conf = target.confidence > best->confidence;
    const bool same_conf_larger = target.confidence == best->confidence && target.area_px > best->area_px;
    if (higher_conf || same_conf_larger) {
      best = target;
    }
  }

  return best;
}

bool ObjectTargetExtractor::IsValidBox(const cv::Rect &box) const {
  return box.width >= cfg_.min_box_width && box.height >= cfg_.min_box_height;
}

ObjectTargetExtractor::Target ObjectTargetExtractor::MakeTarget(const Detection &det) {
  Target target;
  target.class_id = det.class_id;
  target.confidence = det.confidence;
  target.box_px = det.box;
  target.width_px = static_cast<float>(det.box.width);
  target.height_px = static_cast<float>(det.box.height);
  target.area_px = target.width_px * target.height_px;
  target.center_px = cv::Point2f(static_cast<float>(det.box.x) + 0.5f * target.width_px,
                                 static_cast<float>(det.box.y) + 0.5f * target.height_px);
  target.rectified_center_px = target.center_px;
  target.center_rectified = false;
  return target;
}

} // namespace vision
