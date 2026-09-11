#include "vision/line_detection_adapter.hpp"

#include <algorithm>

namespace vision {

LineDetectionAdapter::LineDetectionAdapter(int line_class_id,
                                           float confidence_threshold)
    : line_class_id_(line_class_id),
      confidence_threshold_(confidence_threshold) {}

std::vector<cv::Point2f> LineDetectionAdapter::ExtractCenters(
    const std::vector<Detection> &detections) const {
  std::vector<cv::Point2f> centers;
  centers.reserve(detections.size());
  for (const auto &detection : detections) {
    if (detection.class_id != line_class_id_ ||
        detection.confidence < confidence_threshold_) {
      continue;
    }
    const auto &box = detection.box;
    if (box.width <= 1 || box.height <= 1) {
      continue;
    }
    centers.emplace_back(static_cast<float>(box.x) + 0.5f * box.width,
                         static_cast<float>(box.y) + 0.5f * box.height);
  }
  std::sort(centers.begin(), centers.end(),
            [](const cv::Point2f &a, const cv::Point2f &b) {
              return (a.y == b.y) ? (a.x < b.x) : (a.y > b.y);
            });
  return centers;
}

} // namespace vision
