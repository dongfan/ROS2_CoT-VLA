#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

import numpy as np
import cv2
import pyrealsense2 as rs

from .cot_model import CoTVLAWrapper
from cot_vision_ros2.cot_llava_node import LLaVAClient

class VisionCoTNode(Node):
    def __init__(self):
        super().__init__('vision_cot_node')

        # --- Parameters ---
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('depth_window', 7)
        self.declare_parameter('heat_thresh', 0.6)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 3.5)
        self.declare_parameter('instruction', '주유구를 찾아 열어라')

        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.depth_window = int(self.get_parameter('depth_window').value)
        self.heat_thresh = float(self.get_parameter('heat_thresh').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.instruction = self.get_parameter('instruction').value

        # --- Publishers ---
        qos = QoSProfile(depth=10)
        self.pub_point  = self.create_publisher(PointStamped, '/fuel/object_3d', qos)
        self.pub_action = self.create_publisher(String,       '/cot/action', qos)
        self.pub_viz    = self.create_publisher(Image,        '/fuel/cot_viz', qos)

        # --- NEW: instruction 구독 ---
        self.sub_instruction = self.create_subscription(String, '/fuel/instruction', self.cb_instruction, 10)

        # --- Initialize RealSense ---
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale_rs = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.get_logger().info(f"🧠 CoT-VLA Vision started (RealSense) — depth_scale={self.depth_scale_rs}")

        # --- Load model --- 더미 자료
        self.model = CoTVLAWrapper()
        self.model.warmup()

        self.model = LLaVAClient(mode="http")  # or "ollama" or "dummy"

        self.timer = self.create_timer(1/15.0, self.loop)

        self.current_phase = "cap"
        self.step = 0

    # === NEW ===
    def cb_instruction(self, msg: String):
        new_inst = msg.data.strip()
        if not new_inst:
            return
        self.instruction = new_inst
        self.get_logger().info(f"🧭 Instruction updated → '{new_inst}'")

    # === Main Loop ===
    def loop(self):
        try:
            frames = self.pipe.wait_for_frames()
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale_rs

            out = self.model.infer(color, instruction=self.instruction)
            heat = out.get("heatmap")
            action = out.get("action", "")

            if heat is None:
                return

            # pick target pixel
            mask = heat >= max(self.heat_thresh, float(heat.max()) * 0.9)
            if not mask.any():
                v, u = np.unravel_index(np.argmax(heat), heat.shape)
            else:
                inds = np.transpose(np.nonzero(mask))
                v, u = inds[len(inds)//2]

            # median depth around (u,v)
            win = self.depth_window
            y0, y1 = max(0, v-win//2), min(depth.shape[0], v+win//2+1)
            x0, x1 = max(0, u-win//2), min(depth.shape[1], u+win//2+1)
            patch = depth[y0:y1, x0:x1]
            d = float(np.nanmedian(patch))
            if not (self.min_depth_m <= d <= self.max_depth_m):
                self.get_logger().warn(f"Depth out of range ({u},{v}) {d:.3f} m")
                return

            # simple intrinsics (필요시 CameraInfo 사용으로 교체)
            fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d

            pt = PointStamped()
            pt.header.frame_id = "camera_link"
            pt.point.x, pt.point.y, pt.point.z = float(X), float(Y), float(Z)
            self.pub_point.publish(pt)

            if action:
                self.pub_action.publish(String(data=action))

            # Visualization
            viz = color.copy()
            cv2.circle(viz, (int(u), int(v)), 8, (0,255,0), 2)
            hm = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(viz, 1.0, hm, 0.35, 0)

            msg = Image()
            msg.header.frame_id = "camera_link"
            msg.height, msg.width = vis.shape[:2]
            msg.encoding = 'bgr8'
            msg.step = msg.width * 3
            msg.data = vis.tobytes()
            self.pub_viz.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"[VisionCoTNode] loop error: {e}")

    def reasoning_cycle(self):
        frame = self.get_camera_frame()
        state = self.estimate_state(frame)

        if self.current_phase == "cap":
            if self.step == 0:
                self.publish_action("open_cap_step1")
            elif self.step == 1:
                self.publish_action("open_cap_step2")
            elif self.step == 2:
                self.publish_action("verify_open")

    def destroy_node(self):
        self.pipe.stop()
        super().destroy_node()

def main():
    rclpy.init()
    node = VisionCoTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
