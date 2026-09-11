# 이전 코드 보관소

현재 `vision` 빌드와 실행에는 포함되지 않는 예전 인터페이스를 보관한다.

- `LinePointExtractor`: 점선 detection 중심점 추출만 담당하던 옛 클래스
- `FeatureExtractor`: 중심점의 8차원 특징 계산만 담당하던 옛 클래스
- `CoordinateRectifier`: 공통 코어 호출 전의 OpenCV 좌표 보정 래퍼
- `LineFeatureExtractor`: 중심점 추출과 특징 계산이 합쳐져 있던 래퍼
- `LineVelCmd`: 공통 코어 명령을 ROS `Twist`로 감싸던 래퍼
- `ObjectTargetExtractor`: OpenCV bbox에서 객체 target을 고르던 래퍼
- `BallVelCmd`: ROS `Twist`를 직접 만들던 공 상태·속도제어 래퍼

현재는 `LineDetectionAdapter`가 YOLO detection을 중심점으로 바꾸고,
보정·특징·점선 속도·객체 target 선택·공 속도 계산은
`/home/noh/vision_core`가 직접 담당한다.

이 폴더는 CMake 빌드 및 설치 대상이 아니다. 새 코드에서 이 헤더를 include하지
말고 현재 `vision` 어댑터 또는 독립 `vision_core`를 사용한다.
