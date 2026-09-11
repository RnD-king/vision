# 비전 공통 코어 연결 안내

`/home/noh/vision_core`의 `shared_vision_core`는 실제 ROS2 비전 노드와 G1
시뮬레이터가 함께 사용하는 비전·행동 계산 라이브러리다. 별도 ROS2 노드로
실행하지 않고 각 프로세스 안에서
일반 C++ 함수로 호출되므로 토픽 전송이나 직렬화 비용이 생기지 않는다.

## 담당 범위

- IMU roll/pitch 기반 픽셀 좌표 보정
- 보정된 점 좌표의 8차원 특징 계산
- 일반 점선 추종 속도 계산
- 최근 점선 방향과 경로상 위치를 이용한 점선 누락 복구
- 공·골대·백보드·허들 class별 target 선택
- Line/Ball/Hurdle/Goal 미션 전이와 활성 controller 잠금
- 속도·미션 액션·카메라 요청을 하나의 `ControlCommand`로 취합
- 선택적으로 연속 속도를 P2P 고정 보행 액션으로 양자화

YOLO/TensorRT 추론, ROS2 토픽 구독·발행, 화면 표시는 담당하지 않는다.

## 실제 비전 노드

`line_perception_node`는 `shared_vision_core`를 직접 호출한다. YOLO detection을
점선 중심점과 공통 Detection 자료형으로 바꾸는 변환만 `vision`에 남는다.
`line_detection_adapter`는 라인 bbox를 C++ 중심점 목록으로 바꾸며,
공·골대·백보드·허들은 노드의 `ToCoreDetections()`가 모든 bbox를 한꺼번에
공통 자료형으로 바꿔 코어에 전달한다.
기존 좌표·특징·점선 속도·객체 target·공 속도 래퍼는 `legacy`에 보관한다.

보행 출력은 ROS 파라미터 `locomotion_backend`로 선택한다.

```text
velocity (기본값): /g1_vision/cmd_vel로 연속속도 발행
p2p             : /cmd_vel 발행 없이 /g1_vision/action_cmd로 고정 보행 발행
```

P2P에서도 공 집기·허들 넘기·슛 등 미션 액션과 같은 `action_cmd/status`
ACK/DONE 계약을 사용한다. 보행 DONE은 다음 보행 블록 선택에만 쓰고 미션
controller의 동작 완료로 전달하지 않는다.

```bash
ros2 run vision line_perception_node --ros-args \
  -p locomotion_backend:=p2p
```

실제 노드는 아직 경기장 경로와 로봇 위치를 받지 않으므로, 경로 정보가 필요한
위치 기반 복구는 사용하지 않고 누적된 점선 좌우 방향을 이용한다. 이후 odometry와
경로 입력을 연결하면 같은 코어의 위치 기반 복구를 그대로 사용할 수 있다.

## G1

G1은 `ctypes` 연결층으로 독립 코어의 동일한 공유 라이브러리를 호출한다. 기본값은 기존처럼
카메라 흔들림이 포함된 픽셀을 그대로 사용한다. `use_rp_stabilization=True`로
설정하면 실제 비전 노드와 같은 IMU 좌표 보정을 거친다.

G1과 실제 비전을 실행하기 전에 독립 코어를 먼저 빌드하고 설치해야 한다.

```bash
cd /home/noh/vision_core
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/noh/vision_core/install
cmake --build build -j
cmake --install build

cd /home/noh/my_cv
source /opt/ros/humble/setup.bash
colcon build --packages-select vision --allow-overriding vision
source install/setup.bash
```

공유 라이브러리를 다른 위치에 설치했다면 다음 환경변수로 지정한다.

```bash
export G1_VISION_CORE_LIB=/절대/경로/libshared_vision_core.so
```

## 수정 원칙

전체 구조와 변경 범위별 빌드 방법은 `/home/noh/vision_core/README.md`를
기준으로 한다.

좌표 보정식은 `coordinate_rectifier.cpp`, 특징 정의는
`line_feature_extractor.cpp`, 규칙제어와 복구 계산은
`line_velocity_controller.cpp`, 공 접근은 `ball_controller.cpp`, 미션 잠금은
`mission_controller.cpp`, 최종 명령 생명주기는 `control_command.cpp`, P2P 보행
선택은 `p2p_motion_quantizer.cpp`에서 수정한다. ROS 메시지 변환은 기존 `vision`
클래스에서, Python 자료형 변환은 G1의 `core_bridge.py`에서만 처리한다.

이전에 중심점 추출과 특징 계산을 따로 담당하던 사용 중단 인터페이스는
`src/vision/legacy`에 보관하며 현재 빌드와 설치에는 포함하지 않는다.
