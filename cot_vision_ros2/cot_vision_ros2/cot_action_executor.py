#!/usr/bin/env python3
import math
import json
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration

# TF
import tf2_ros
import tf2_geometry_msgs  # noqa: F401

# (선택) Doosan ROS2 서비스가 있으면 import, 없으면 무시
try:
    from dsr_msgs2.srv import MoveJoint, MoveLine
    from dsr_msgs2.srv import SetGpioAnalog, SetGpioDigital
    DOOSAN_OK = True
except Exception:
    DOOSAN_OK = False


class CotActionExecutor(Node):
    """
    - /cot/action          : "move_to_target" | "grasp_nozzle" | "open_cap"
    - /planner/object_3d_refined (PointStamped, camera frame)  # cot_llava_node가 발행:contentReference[oaicite:4]{index=4}
    - /fuel/object_3d              (PointStamped, camera/camera_link)  # vision_cot_node가 발행:contentReference[oaicite:5]{index=5}

    동작:
      1) 가장 최근 3D 지점을 base로 TF 변환
      2) action에 따라 접근-접촉-수행 시퀀스 실행
    """

    def __init__(self):
        super().__init__("cot_action_executor")

        # --- Parameters ---
        self.declare_parameter("camera_frame", "camera_color_optical_frame")  # cot_llava_node 기본값:contentReference[oaicite:6]{index=6}
        self.declare_parameter("fallback_camera_frame", "camera_link")        # vision_cot_node 기본값:contentReference[oaicite:7]{index=7}
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("tool_down_rpy_deg", [180.0, 0.0, 90.0])  # (R,P,Y) 도슨 관례 상단지향 예시
        self.declare_parameter("approach_offset_m", 0.08)   # 수직 접근 오프셋
        self.declare_parameter("touch_down_m",     0.03)    # 마지막 접촉 내리기
        self.declare_parameter("open_cap_angle_deg", -90.0) # 캡 돌리기 각도
        self.declare_parameter("vel", 30.0)
        self.declare_parameter("acc", 60.0)

        self.camera_frame = self.get_parameter("camera_frame").value
        self.camera_frame_fallback = self.get_parameter("fallback_camera_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.rpy_deg = self.get_parameter("tool_down_rpy_deg").value
        self.approach_offset = float(self.get_parameter("approach_offset_m").value)
        self.touch_down = float(self.get_parameter("touch_down_m").value)
        self.open_cap_angle = float(self.get_parameter("open_cap_angle_deg").value)
        self.vel = float(self.get_parameter("vel").value)
        self.acc = float(self.get_parameter("acc").value)

        # --- Subscriptions ---
        self.sub_action = self.create_subscription(String, "/cot/action", self.on_action, 10)
        # cot_llava_node가 주는 정제 좌표(우선시):contentReference[oaicite:8]{index=8}
        self.sub_pt_refined = self.create_subscription(PointStamped, "/planner/object_3d_refined", self.on_point, 10)
        # vision_cot_node가 주는 좌표(백업):contentReference[oaicite:9]{index=9}
        self.sub_pt_fuel    = self.create_subscription(PointStamped, "/fuel/object_3d", self.on_point, 10)

        # --- TF Buffer/Listener ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- (선택) Doosan 서비스 클라이언트 준비 ---
        if DOOSAN_OK:
            self.cli_movej = self.create_client(MoveJoint, "/dsr01/motion/move_joint")
            self.cli_movel = self.create_client(MoveLine,  "/dsr01/motion/move_line")
            # 그리퍼/IO는 장비마다 다르므로 예시로만 둡니다.
        else:
            self.get_logger().warn("Doosan ROS2 서비스 미탑재로 '로그 시뮬레이션' 모드로 실행합니다.")

        self.last_point_cam: Optional[PointStamped] = None
        self.get_logger().info("🤖 CoT Action Executor ready (Python)")

    # ========== Callbacks ==========
    def on_point(self, msg: PointStamped):
        """최근 카메라 좌표 갱신"""
        self.last_point_cam = msg

    def on_action(self, msg: String):
        action = msg.data.strip()
        self.get_logger().info(f"🧠 /cot/action = {action}")

        p_base = self.get_target_in_base()
        if p_base is None:
            self.get_logger().warn("타깃 좌표 없음(또는 TF 변환 실패) → 동작 스킵")
            return

        if action == "move_to_target":
            self.seq_move_to_target(p_base)
        elif action == "grasp_nozzle":
            self.seq_grasp_nozzle(p_base)
        elif action == "open_cap":
            self.seq_open_cap(p_base)
        else:
            self.get_logger().info(f"알 수 없는 action='{action}' → 기본 이동만 수행")
            self.seq_move_to_target(p_base)

    # ========== Helpers ==========
    def get_target_in_base(self) -> Optional[PoseStamped]:
        """최근 카메라 좌표를 base 프레임으로 변환하고 도구 방향을 부여"""
        if self.last_point_cam is None:
            return None

        src = PointStamped()
        src.header = self.last_point_cam.header
        src.point = self.last_point_cam.point

        from_frames = [src.header.frame_id, self.camera_frame, self.camera_frame_fallback]
        tf_ok = None
        for frm in from_frames:
            try:
                out = self.tf_buffer.transform(
                    src, self.base_frame, timeout=Duration(seconds=0.2)
                )
                tf_ok = out
                break
            except Exception:
                # frame 이름이 다를 수 있어 순차 시도
                src.header.frame_id = frm

        if tf_ok is None:
            self.get_logger().warn(f"TF 변환 실패: tried {from_frames} → {self.base_frame}")
            return None

        # 위치 + 고정 RPY(툴 하향)
        r, p, y = [math.radians(d) for d in self.rpy_deg]
        pose = PoseStamped()
        pose.header = tf_ok.header
        pose.header.frame_id = self.base_frame
        pose.pose.position.x = tf_ok.point.x
        pose.pose.position.y = tf_ok.point.y
        pose.pose.position.z = tf_ok.point.z

        # RPY → quaternion
        qw, qx, qy, qz = rpy_to_quat(r, p, y)
        pose.pose.orientation.w = qw
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        return pose

    # ========== Sequences ==========
    def seq_move_to_target(self, target: PoseStamped):
        self.get_logger().info("🚶 move_to_target: 접근→접촉")
        approach = self.offset_pose(target, dz=self.approach_offset)
        self.movej_or_log(approach, desc="approach")
        contact  = self.offset_pose(target, dz=self.touch_down)
        self.movel_or_log(contact,  desc="touch_down")

    def seq_grasp_nozzle(self, target: PoseStamped):
        self.get_logger().info("🫳 grasp_nozzle: 접근→집기")
        self.seq_move_to_target(target)
        self.gripper_close_or_log(force=30)
        self.get_logger().info("✅ nozzle grasped")

    def seq_open_cap(self, target: PoseStamped):
        self.get_logger().info("🔧 open_cap: 캡 위치로 접근→회전")
        self.seq_move_to_target(target)
        self.compliance_rotate_or_log(angle_deg=self.open_cap_angle)
        self.get_logger().info("✅ cap opened")

    # ========== Low-level wrappers ==========
    def movej_or_log(self, pose: PoseStamped, desc=""):
        if DOOSAN_OK and self.cli_movej.wait_for_service(timeout_sec=0.1):
            req = MoveJoint.Request()
            # 실제 dsr_msgs2.srv 정의에 맞춰 포맷 수정 필요(예: posx, ref, vel, acc, time 등)
            # 여기서는 예시로 Pose만 기록
            self.get_logger().info(f"[movej] {desc} → ({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})")
            # TODO: req 채우기 및 self.cli_movej.call_async(req)
        else:
            self.get_logger().info(f"[SIM] movej {desc} → {pose_summary(pose)}")

    def movel_or_log(self, pose: PoseStamped, desc=""):
        if DOOSAN_OK and self.cli_movel.wait_for_service(timeout_sec=0.1):
            req = MoveLine.Request()
            self.get_logger().info(f"[movel] {desc} → ({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})")
            # TODO: req 채우기 및 self.cli_movel.call_async(req)
        else:
            self.get_logger().info(f"[SIM] movel {desc} → {pose_summary(pose)}")

    def gripper_close_or_log(self, force: int = 20):
        # 실제 그리퍼 드라이버/IO에 맞춰 바꾸세요.
        self.get_logger().info(f"[SIM] gripper_close (force={force})")

    def compliance_rotate_or_log(self, angle_deg: float):
        # 실제 순응/토크 제어 API에 맞춰 바꾸세요 (예: task_compliance_ctrl)
        self.get_logger().info(f"[SIM] task_compliance_ctrl rotate {angle_deg:.1f} deg")

    # ========== Utils ==========
    @staticmethod
    def offset_pose(p: PoseStamped, dx=0.0, dy=0.0, dz=0.0) -> PoseStamped:
        out = PoseStamped()
        out.header = p.header
        out.pose = p.pose
        out.pose.position.x += dx
        out.pose.position.y += dy
        out.pose.position.z += dz
        return out


def rpy_to_quat(r, p, y):
    """roll-pitch-yaw → quaternion(w,x,y,z)"""
    cy = math.cos(y * 0.5); sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5); sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5); sr = math.sin(r * 0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qw, qx, qy, qz


def pose_summary(p: PoseStamped) -> str:
    return json.dumps({
        "frame": p.header.frame_id,
        "pos": [round(p.pose.position.x, 4), round(p.pose.position.y, 4), round(p.pose.position.z, 4)],
    })


def main():
    rclpy.init()
    node = CotActionExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
