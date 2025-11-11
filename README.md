# ROS2_CoT-VLA
📘 프로젝트 개요

이 프로젝트는 Chain-of-Thought Vision–Language–Action (CoT-VLA) 구조를 로봇 제어 파이프라인에 적용하여,
카메라 입력 → 시각-언어 추론 → 3D 타깃 추정 → 로봇팔 동작 명령을 자동으로 연결하는 지능형 주유 로봇 시스템입니다.

CoT-VLA는 기존의 단순 Vision-Action 매핑과 달리,
“시각적 추론(Vision Reasoning) + 언어적 명령 해석(Language Understanding) + 행동 의도(Action Planning)”의
세 단계를 연결하여 보다 유연한 로봇 의사결정을 가능하게 합니다.

🔍 주요 구성 모듈
1️⃣ vision_cot_node.py — CoT-VLA 시각 인식 노드

vision_cot_node

RealSense 카메라로부터 RGB + Depth 프레임을 수신.

내부 LLaVAClient 또는 CoTVLAWrapper 모델을 이용해 “heatmap + action”을 추론.

가장 높은 확률의 픽셀을 선택해 3D 좌표(PointStamped) 로 변환하여 /fuel/object_3d 토픽 발행.

예시 액션:

"move_to_target"

"grasp_nozzle"

"open_cap"

/fuel/instruction 토픽을 구독해 동적으로 명령 변경 가능.

2️⃣ cot_llava_node.py — LLaVA 멀티모달 모델 연동

cot_llava_node

LLaVA-1.6 계열 모델을 dummy / http / ollama 모드 중 선택 실행.

/cot/instruction + /camera 입력을 받아
heatmap + action 형태의 결과를 /cot/action, /planner/object_3d_refined 로 퍼블리시.

HTTP 모드에서는 OpenAI-호환 REST API (LLAVA_API_URL, LLAVA_API_KEY)로 연결.
Ollama 모드는 로컬 멀티모달 모델(llava:13b 등) 사용 가능.

3️⃣ cot_action_executor.py — 로봇 제어 실행기

cot_action_executor

/cot/action 명령 및 /planner/object_3d_refined 또는 /fuel/object_3d 좌표를 구독.

TF 변환을 통해 카메라 좌표를 Doosan E0509 base 프레임으로 변환.

각 액션별 시퀀스를 수행:

move_to_target → 접근-접촉 이동

grasp_nozzle → 그리퍼 닫기

open_cap → 순응 제어를 통한 회전

실제 dsr_msgs2.srv.MoveJoint, MoveLine 서비스 호출 혹은 시뮬레이션 로그 모드로 동작.

4️⃣ cot_model.py — CoT-VLA Wrapper Stub

cot_model

간단한 heatmap + action 추론 모형 (중심 Gaussian 기반 더미).

실제 CoT-VLA 모델(예: CoT-VLA-7B, LLaVA-RL, BLIP-VLA 등)로 교체 가능.

5️⃣ camera_geom.py — 카메라 내참수 변환 유틸

camera_geom

PinholeIntrinsics 구조체를 통해 fx, fy, cx, cy 내참수를 정리.

deproject_pixel_to_camera(u,v,Z) 로 픽셀 좌표를 카메라 좌표로 변환.

6️⃣ tf_utils.py — TF 변환 래퍼

tf_utils

TFBufferProxy.transform_point()
→ rclpy TF2 버퍼를 이용한 안전한 포인트 변환 지원.

7️⃣ license_plate_reader.py — 번호판 인식

license_plate_reader

OpenCV + Tesseract OCR 사용.

/camera/color/image_raw → /fuel/license_plate 발행.

최초 한 번만 음성 안내(gTTS) 실행.

8️⃣ speech_feedback.py — 음성 피드백 FSM

speech_feedback

/fuel/license_plate, /cot/action, /fuel/cap_opened 이벤트를 수신.

주유 단계별 상태 기반 안내:

차량 인식 → “주유 시작하겠습니다.”

탐색 중 → “주유구를 찾는 중입니다.”

탐색 성공 → “주유구를 열어주세요.”

캡 열림 → “주유구를 열겠습니다.”

/cot/action 토픽을 통해 로봇에 직접 명령 전송.

🚀 실행 방법
# ROS2 환경 설정
source /opt/ros/humble/setup.bash
source ~/xyz_ws/install/setup.bash

# Vision-CoT (카메라/추론)
ros2 run cot_vision_ros2 vision_cot_node

# LLaVA Node (멀티모달 추론)
ros2 run cot_vision_ros2 cot_llava_node

# Action Executor (Doosan 제어)
ros2 run cot_vision_ros2 cot_action_executor

# 음성 피드백 + 번호판 인식
ros2 run cot_vision_ros2 speech_feedback
ros2 run cot_vision_ros2 license_plate_reader


Chain-of-Thought (CoT) 논리를 통해,
단순 분류형 모델이 아니라 *“명령의 의미를 해석하고 그에 따라 행동을 선택”*하도록 합니다.
→ 예: “노즐을 잡아라” → grasp_nozzle → 3D 좌표 선택 → 로봇팔 이동 및 집기 수행.

🧱 확장 및 연구 포인트

🤖 실세계 강화학습 적용: CoT-VLA 출력을 reward로 활용한 강화학습 fine-tuning.

🌐 멀티모달 LLM 연동: LLaVA-Next, Qwen-VL-Max, InternVL 등과 교체 가능.

🦾 로봇 시뮬레이션 연동: Isaac Sim 5.x 또는 Unity XR 환경으로의 확장.

🧩 다중 객체 추론: heatmap 기반 attention pooling → multi-target tasking.