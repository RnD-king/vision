#pragma once

// 사용 중단: 현재는 vision/line_detection_adapter.hpp를 사용한다.

#include <opencv2/core.hpp>
#include <vector>

#include "vision/types.hpp"

namespace vision {

class LinePointExtractor {
public:
  explicit LinePointExtractor(int line_class_id, float conf_thres);
  std::vector<cv::Point2f>
  ExtractCenters(const std::vector<Detection> &dets) const;

private:
  int line_class_id_{0};
  float conf_thres_{0.45f};
};

} // namespace vision
