uint8 MISSION_LINE=1
uint8 MISSION_BALL=2
uint8 MISSION_GOAL=3
uint8 MISSION_HURDLE=4

uint16 DEFAULT_POSITION=1
uint16 DEFAULT_POSE_MODE=2
uint16 STEP_FORWARD_HALF=3
uint16 STEP_BACK=4
uint16 LEFT_SIDE_STEP=5
uint16 RIGHT_SIDE_STEP=6
uint16 TURN_LEFT=7
uint16 TURN_RIGHT=8
uint16 WALK_MODE=9
uint16 STEP_FORWARD_ONE=10
uint16 STEP_FORWARD_LEFT=11
uint16 STEP_FORWARD_RIGHT=12
uint16 STEP_FORWARD_FIVE=13
uint16 PICK_BALL=14
uint16 RECATCH=15
uint16 HUDDLE=16
uint16 SHOOT=17
uint16 TURN_LEFT_AND_STEP=18
uint16 TURN_RIGHT_AND_STEP=19

uint64 action_id
uint8 mission
uint16 action
# SHOOT은 좌회전(+)/우회전(-) signed degree를 사용한다.
# TURN_LEFT/RIGHT(±_AND_STEP 포함)는 action으로 방향을 구분하고
# target_yaw_deg에는 회전량의 양수 정수 degree만 전달한다.
# 그 외 action에서는 0이다.
int16 target_yaw_deg
