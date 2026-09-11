#pragma once

// YOLO postprocess 이후 vision_core 모듈들이 공유하는 기본 detection 타입.

#include <opencv2/core.hpp>

namespace vision {

struct Detection {
  cv::Rect box;
  float confidence{0.0f};
  int class_id{0};
};

} // namespace vision
