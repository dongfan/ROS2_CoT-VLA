#!/usr/bin/env python3
"""
MotionController (CoT-VLA 최종판)
- 단계별 instruction 자동 전환
- CoT-VLA의 /fuel/cot_action 수신 → 실제 로봇/그리퍼 제어
- /fuel/object_3d 좌표로 접근 (cap / nozzle 모드 모두 재사용)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Bool

import numpy as np
from collections import deque
import time

import DR_init
from smartfuel_core.gripper_drl_controller import GripperController

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# 카메라 ↔ TCP 간 대략 오프셋 (필요 시 조정)
CAMERA_OFFSET_TCP_Z_M = 0.05
# -Y(바닥 방향) 기본 프리셋
ORIENT_PRESET_POSJ = (20, 35, 105, 105, -90, 50)

# 단계별 instruction 문구
INSTR_CAP    = "물병을 찾아 뚜껑을 열어라"
INSTR_NOZZLE = "노즐을 찾아 잡아라"
INSTR_INSERT = "노즐을 주유구에 꽂아라"

class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')
        self.get_logger().info("🤖 MotionController (CoT-VLA) starting...")

        # 상태/버퍼
        self.coord_buffer = deque(maxlen=10)
        self.last_base_coords = None
        self.is_busy = False
        self.mode = "fuel_cap"     # fuel_cap / nozzle
        self.phase = "cap"         # cap → nozzle → insert → done

        # 그리퍼 초기화
        self._init_gripper_and_home()

        # 퍼블리셔/구독
        self.sub_obj3d  = self.create_subscription(PointStamped, '/fuel/object_3d', self.object_callback, 10)
        self.sub_action = self.create_subscription(String, '/fuel/cot_action',  self.on_cot_action, 10)
        self.sub_stop   = self.create_subscription(Bool,   '/stop_motion',      self.on_stop_signal, 10)
        self.pub_instruction = self.create_publisher(String, '/fuel/instruction', 10)

        self.get_logger().info("✅ Ready: /fuel/object_3d, /fuel/cot_action, /fuel/instruction")

        # 초기 Hand–Eye 행렬
        self.T_tcp2cam = self._make_tcp2cam_matrix(self.mode)

        # 주기 제어(필요 시 사용)
        self.timer = self.create_timer(0.5, self.control_loop)

        # 테스트 모드: 결제 없이 -Y 정렬 → 첫 단계 instruction 발행
        self.get_logger().info("🧪 테스트 모드: 결제신호 없이 -Y 정렬 후 단계 시작")
        self.orient_negative_y()
        self.set_handeye_mode("fuel_cap")
        self.next_step()   # phase="cap" → INSTR_CAP 발행

    # ───────────────────────── 기본 유틸 ─────────────────────────
    def _init_gripper_and_home(self):
        try:
            from DSR_ROBOT2 import wait, movej
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            if not self.gripper.initialize():
                raise RuntimeError("Gripper initialization failed")
            self.gripper.move(0)
            wait(1.0)
            movej([0, 0, 90, 0, 90, 0], 80, 80)
            wait(1.0)
        except Exception as e:
            self.get_logger().error(f"❌ Gripper/Init error: {e}")
            raise

    def _make_tcp2cam_matrix(self, mode: str):
        T = np.eye(4)
        if mode == "fuel_cap":
            T[:3, :3] = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        elif mode == "nozzle":
            T[:3, :3] = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        T[:3, 3] = [0, 0, CAMERA_OFFSET_TCP_Z_M]
        return T

    def set_handeye_mode(self, mode: str):
        if mode not in ("fuel_cap", "nozzle"):
            self.get_logger().warn(f"⚠️ Unknown hand-eye mode: {mode}")
            return
        self.mode = mode
        self.T_tcp2cam = self._make_tcp2cam_matrix(mode)
        self.get_logger().info(f"🔁 Hand–Eye 모드 변경: {mode}")

    def pose_to_matrix(self, pose):
        x, y, z, rx, ry, rz = pose
        rx, ry, rz = np.deg2rad([rx, ry, rz])
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        R = Rz @ Ry @ Rx
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = [x/1000.0, y/1000.0, z/1000.0]
        return T
    
    # ───────────────────────── 좌표 noise 제거용 ─────────────────────────
    def smooth_coordinates(self, Xb, Yb, Zb):
        """최근 좌표 평균을 통한 이동평균 필터"""
        if not hasattr(self, "coord_buffer"):
            from collections import deque
            self.coord_buffer = deque(maxlen=10)
        self.coord_buffer.append((Xb, Yb, Zb))
        if len(self.coord_buffer) < 3:
            return Xb, Yb, Zb
        avg = np.mean(self.coord_buffer, axis=0)
        return avg[0], avg[1], avg[2]

    def filter_jump(self, Xb, Yb, Zb, threshold=0.05):
        """좌표 점프 방지: 이전 좌표 대비 급격한 변화 제거"""
        if not hasattr(self, "last_valid_coord") or self.last_valid_coord is None:
            self.last_valid_coord = (Xb, Yb, Zb)
            return Xb, Yb, Zb
        Xp, Yp, Zp = self.last_valid_coord
        if (abs(Xb - Xp) > threshold or
            abs(Yb - Yp) > threshold or
            abs(Zb - Zp) > threshold):
            self.get_logger().warn("⚠️ 좌표 점프 감지 → 이전 좌표 유지")
            return Xp, Yp, Zp
        self.last_valid_coord = (Xb, Yb, Zb)
        return Xb, Yb, Zb
    
    # ───────────────────────── 좌표 수신 ─────────────────────────
    def object_callback(self, msg: PointStamped):
        """YOLO 3D 포인트 수신 → Base 좌표 변환 및 저장"""
        from DSR_ROBOT2 import get_current_posx

        try:
            Xc, Yc, Zc = msg.point.x, msg.point.y, msg.point.z  # meter 단위
            pose = get_current_posx()[0][:6]
            T_base2tcp = self.pose_to_matrix(pose)
            T_tcp2cam = self.T_tcp2cam
            T_base2cam = T_base2tcp @ T_tcp2cam

            cam_point = np.array([[Xc], [Yc], [Zc], [1]])
            base_point = T_base2cam @ cam_point
            Xb, Yb, Zb = base_point[:3, 0]  # meter 그대로

            # 📉 필터 적용
            Xb, Yb, Zb = self.smooth_coordinates(Xb, Yb, Zb)
            Xb, Yb, Zb = self.filter_jump(Xb, Yb, Zb)
            self.last_base_coords = (Xb, Yb, Zb)

            self.get_logger().info(f"📍 감지 좌표(Base): X={Xb:.3f} Y={Yb:.3f} Z={Zb:.3f}")

            # 탐색 종료 트리거
            if getattr(self, "searching", False):
                self.get_logger().info("🛑 감지됨 → 탐색 종료 후 이동 준비")
                self.searching = False
                self.ready_to_move = True

        except Exception as e:
            self.get_logger().warn(f"⚠️ 좌표 변환 실패, 재시도 예정: {e}")
            self.ready_to_move = False

    # ───────────────────────── 제어 루프(옵션) ───────────────────
    def control_loop(self):
        # 필요 시 주기 동작이 있으면 사용 (현재는 on_cot_action 이벤트 구동이 메인)
        return

    # ───────────────────────── 액션 수신 → 실제 동작 ────────────
    def on_cot_action(self, msg: String):
        act = msg.data.strip().lower()
        self.get_logger().info(f"🧠 [CoT-VLA] Action received: {act}")
        
        if act == "orient_minus_y":
            # 툴 방향 정렬 (-Y 축)
            self.get_logger().info("🧭 툴을 -Y 방향으로 회전합니다.")
            self.orient_negative_y()
            # 선택적으로 상태 업데이트
            self.phase = "search"
            self.next_step()

        elif act == "search_fuel_port":
            # Vision 탐색 시작
            self.get_logger().info("🔎 주유구 탐색 모드로 전환합니다.")
            self.start_vision_tracking()   # vision_cot_node 실행 or 토픽 신호
            self.phase = "detect"
            self.next_step()
    
        elif act == "open_cap":
            # 주유구 열기 (그리퍼로 잡고 회전)
            if not self.wait_for_detection(timeout_sec=15.0):
                return
            self.set_handeye_mode("fuel_cap")
            self._approach_hold(axis="y", hold_mm=170.0)   # 주유구 앞 -Y축쪽 스탠드오프
            self._wrist_align_for_cap()
            self.rotate_grip(2, True)
            # 다음 단계로
            self.phase = "nozzle"
            self.next_step()
            
        elif act == "grasp_nozzle":
            # 노즐 잡기
            self.set_handeye_mode("nozzle")
            self._approach_hold(axis="x", hold_mm=80.0)    # 노즐 앞 +X축쪽 스탠드오프
            self.gripper.move(600)
            # 다음 단계로
            self.phase = "insert"
            self.next_step()

        elif act == "insert_nozzle":
            # 노즐 삽입 (간단 버전: 타깃 방향으로 조금 전진)
            self._insert_motion()
            self.phase = "done"
            self.next_step()

        elif act == "release":
            self.gripper.move(0)

        else:
            self.get_logger().warn(f"⚠️ Unknown action: {act}")

    def start_vision_tracking(self):
        """Vision 노드 (주유구 탐색) 활성화 트리거"""
        from std_msgs.msg import String
        self.get_logger().info("👁️ Vision 탐색 노드에 탐색 지시 신호 전송")
        pub = self.create_publisher(String, '/fuel/search_mode', 10)
        pub.publish(String(data="start"))
        
    # ───────────────────────── 단계 전환/인스트럭션 ────────────
    def next_step(self):
        """현재 phase에 맞춰 CoT-VLA에 instruction 전송"""
        if self.phase == "cap":
            self.publish_instruction(INSTR_CAP)
        elif self.phase == "nozzle":
            self.publish_instruction(INSTR_NOZZLE)
        elif self.phase == "insert":
            self.publish_instruction(INSTR_INSERT)
        elif self.phase == "done":
            self.get_logger().info("🏁 전체 시퀀스 완료")
        else:
            self.get_logger().warn(f"⚠️ Unknown phase: {self.phase}")

    def publish_instruction(self, text: str):
        msg = String(); msg.data = text
        self.pub_instruction.publish(msg)
        self.get_logger().info(f"🧭 [CoT-VLA] Instruction 전송 → \"{text}\"")

    # ───────────────────────── 보조 동작들 ─────────────────────
    def wait_for_detection(self, timeout_sec=10.0):
        """주유구(또는 노즐) 좌표를 기다림 — VisionCoTNode가 /fuel/object_3d를 퍼블리시할 때까지"""
        start = time.time()
        self.get_logger().info(f"⏳ 객체 감지 대기 중 (최대 {timeout_sec:.1f}s)...")
        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.last_detected_point is not None:
                self.get_logger().info("✅ 객체 감지됨 — 이동 준비 완료")
                return True
        self.get_logger().warn("⏰ 감지 실패 — 탐색으로 전환합니다.")
        self.search_for_object()  # 기존 탐색 루틴 호출
        return False

    def _approach_hold(self, axis: str, hold_mm: float):
        """현재 모드에서 마지막 타깃 좌표 기준으로 스탠드오프 위치로 접근"""
        if not self.last_base_coords:
            self.get_logger().warn("⏸ 타깃 좌표 없음 → 접근 보류")
            return
        from DSR_ROBOT2 import movel, wait, get_current_posx, DR_MV_MOD_ABS
        from DR_common2 import posx
        Xb, Yb, Zb = self.last_base_coords
        pose = get_current_posx()[0][:6]

        tx, ty, tz = Xb*1000, Yb*1000, Zb*1000
        if axis.lower() == "y":
            ty = ty + hold_mm
        elif axis.lower() == "x":
            tx = tx - hold_mm

        target = posx(tx, ty, tz, pose[3], pose[4], pose[5])
        movel(target, v=30, a=30, mod=DR_MV_MOD_ABS)
        wait(1.0)

    def _wrist_align_for_cap(self):
        """캡을 비틀기 좋은 손목자세(간단 회전 예)"""
        from DSR_ROBOT2 import movel, DR_MV_MOD_REL, wait
        from DR_common2 import posx
        movel(posx(0, 0, 0, 0, 45, 0), v=50, a=50, mod=DR_MV_MOD_REL)
        wait(0.8)

    def _insert_motion(self):
        """노즐 삽입(간단 전진) — 필요 시 컴플라이언스/힘제어로 확장"""
        from DSR_ROBOT2 import movel, DR_MV_MOD_REL, wait
        from DR_common2 import posx
        movel(posx(0, -40, 0, 0, 0, 0), v=20, a=20, mod=DR_MV_MOD_REL)  # -Y로 40mm 전진 (예시)
        wait(1.0)

    def rotate_grip(self, cnt: int, b_open: bool = True):
        from DSR_ROBOT2 import movej, wait, DR_MV_MOD_REL
        from DR_common2 import posj
        count = 0
        delta = -120 if b_open else 120
        while count < cnt:
            self.gripper.move(400); wait(1.0)
            movej(posj(0,0,0,0,0,delta), v=120, a=120, mod=DR_MV_MOD_REL)
            wait(0.8)
            count += 1
            if count < cnt:
                self.gripper.move(150); wait(0.8)
                movej(posj(0,0,0,0,0,-delta), v=120, a=120, mod=DR_MV_MOD_REL)
                wait(0.6)

    def orient_negative_y(self):
        from DSR_ROBOT2 import movej, wait, DR_MV_MOD_ABS
        from DR_common2 import posj
        self.get_logger().info("🧭 툴을 -Y(바닥) 방향으로 회전 중…")
        movej(posj(*ORIENT_PRESET_POSJ), v=50, a=50, mod=DR_MV_MOD_ABS)
        wait(2.0)
        self.get_logger().info("✅ -Y 정렬 완료")

    def on_stop_signal(self, msg: Bool):
        if msg.data:
            self.get_logger().warn("🛑 Stop requested (no hard-stop here)")

def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    node = MotionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 MotionController stopped.")
    finally:
        try:
            if hasattr(node, "gripper") and node.gripper:
                node.gripper.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
