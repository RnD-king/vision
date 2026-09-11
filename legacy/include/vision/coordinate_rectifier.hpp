#pragma once

// 사용 중단: 현재 line_perception_node는 vision_core/coordinate_rectifier.hpp를
// 직접 호출한다.

// Pipeline step 6 interface.
// line 중심점 배열과 object target 중심점을 같은 방식으로 픽셀 보정할 때 쓴다.

#include <opencv2/core.hpp>
#include <vector>

namespace vision {

class CoordinateRectifier {
public:
  struct Intrinsics {
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
  };

  explicit CoordinateRectifier(const Intrinsics &K);

  // 입력: 픽셀 좌표(u,v) 리스트 + IMU roll/pitch (rad)
  // 출력: roll/pitch 영향을 제거한 "수평 카메라 가정"의 픽셀 좌표(u',v')
  //
  // 중요(명시적 가정):
  // - 카메라 좌표계: x=오른쪽, y=아래, z=전방(일반적인 이미지 좌표계와 일치)
  // - roll  : 카메라 전방축(z) 기준 회전(이미지가 좌/우로 기우는 성분)
  // - pitch : 카메라 우측축(x) 기준 회전(이미지가 위/아래로 숙여지는 성분)
  //
  // 만약 실제 IMU 정의가 이와 다르면(축/부호/순서가 다르면),
  // 출력이 "원하는 방향으로 평탄화"되지 않으므로 해당 가정에 맞게 축/부호만
  // 수정하면 된다.
  std::vector<cv::Point2f>
  RectifyPixelPoints(const std::vector<cv::Point2f> &pts_px, double roll_rad,
                     double pitch_rad) const;

private:
  Intrinsics K_;
};

} // namespace vision
