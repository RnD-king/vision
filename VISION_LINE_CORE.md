# Vision P2P control contract

`vision`은 카메라·TensorRT·ROS 전송만 담당하고, perception과 mission FSM은
ROS 비의존 `vision_core`의 `MissionController`가 단일하게 결정한다.

## 출력

- 보행과 특수 동작은 `/jandi_vision/action_cmd`의 기존 action ID를 사용한다.
- `CONTACT_WALK=20`은 protocol에만 예약되어 있다. 현재 HURDLE contact는
  executor mapping이 준비될 때까지 `STEP_FORWARD_ONE=10`을 사용한다.
- TURN action의 `target_yaw_deg`는 양수 크기이며 방향은 action ID가 나타낸다.
- SHOOT의 `target_yaw_deg`만 좌회전 양수·우회전 음수인 signed degree다.
- ACK/READY/DONE과 `action_id`, LINE one-action READY queue 계약은 유지한다.

## LINE

정상 LINE은 near fit의 offset/heading score만으로
`STEP_FORWARD_LEFT/FIVE/RIGHT`를 직접 고른다. curvature는 별도 validity와
분모로 집계되는 진단값이며 조향이나 미션 전환에 사용하지 않는다.

첫 판단 실패는 2초 정지 관측한다. 유효 O/H 표본이 5개 이상이면 평균으로
복귀한다. 부족하면 안정된 과거 방향 기억이 있을 때 같은 방향으로 15도
회전한 뒤 다시 2초 관측하며 최대 5회 반복한다. 방향 기억이 없으면 회전 없이
2초 관측을 최대 5회 추가한다. 한도를 넘으면 Reset 전까지 FINAL HOLD다.

긴 LINE action 동안 후반 50% 표본을 끝으로 갈수록 1→3 가중 집계하며,
READY 예약 action도 최초 action과 같은 direct selector로 결정한다.

## 객체 미션

- BALL: 원거리 3-way direct cruise, 카메라 DOWN 후 lateral 우선 fine 보행,
  거리 보정 뒤 PICK_BALL, 성공 시 한 번 후진하고 LINE을 재획득한다.
- HURDLE: 원거리 `STEP_FORWARD_FIVE`, raw v 0.75의 10-window/7-hit close
  trigger, 현재 action DONE 후 카메라 DOWN → action 10 → HUDDLE(16).
- GOAL: 백보드 RGB-D로 림 중심 geometry를 매 fine step 뒤 다시 계산한다.
  거리 보정을 먼저 하고, shoot yaw가 ±30도 안이면 signed SHOOT, 아니면
  좌우 side step 후 재관측한다.

카메라 trigger는 실행 중 보행을 취소하지 않는다. trigger만 latch하고 보행
DONE 뒤 camera command를 발행하며, camera settled 전에는 새 보행을 만들지 않는다.

공통 수치 기본값은 `vision_core/config/vision_algorithm.yaml` 하나만 사용한다.
ROS 어댑터 설정은 `config/vision_params.yaml`에 둔다.
