// 사용 중단: vision_core로 이식되기 전의 ROS/OpenCV 어댑터 구현.
//
// Pipeline step 6.
// IMU roll/pitch로 픽셀 중심점을 수평 카메라 기준처럼 보정한다.
// line에는 여러 중심점 전체를 적용하고, object target에는 중심점만 적용한다.
// object bbox width/height/area는 접근/정지 판단용 원본 값으로 유지한다.
// assume_zero_imu:=true 또는 use_imu_rectification:=false면 사실상 영향 없다.

#include "vision/coordinate_rectifier.hpp"
#include "vision_core/coordinate_rectifier.hpp"

#include <cmath>

namespace vision {

CoordinateRectifier::CoordinateRectifier(
    const Intrinsics &K) // 라디안으로 받아오는 중
    : K_(K) {}

std::vector<cv::Point2f>
CoordinateRectifier::RectifyPixelPoints(const std::vector<cv::Point2f> &pts_px,
                                        double roll_rad,
                                        double pitch_rad) const {
  std::vector<vision_core::Point2> input;
  input.reserve(pts_px.size());
  for (const auto &point : pts_px) input.push_back({point.x, point.y});
  const auto rectified = vision_core::RectifyPixelPoints(
      input, {K_.fx, K_.fy, K_.cx, K_.cy}, roll_rad, pitch_rad);
  std::vector<cv::Point2f> out;
  out.reserve(rectified.size());
  for (const auto &point : rectified) {
    out.emplace_back(static_cast<float>(point.u), static_cast<float>(point.v));
  }
  return out;
}

} // namespace vision
