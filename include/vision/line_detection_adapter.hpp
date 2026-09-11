#pragma once

// ROS/OpenCV 쪽 YOLO detection을 공통 코어 입력인 점선 중심점으로 바꾸는 어댑터.

#include <opencv2/core.hpp>
#include <vector>

#include "vision/types.hpp"

namespace vision {

class LineDetectionAdapter {
public:
  LineDetectionAdapter(int line_class_id, float confidence_threshold);

  std::vector<cv::Point2f>
  ExtractCenters(const std::vector<Detection> &detections) const;

private:
  int line_class_id_{0};
  float confidence_threshold_{0.60f};
};

} // namespace vision
