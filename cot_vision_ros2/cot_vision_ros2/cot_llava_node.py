#!/usr/bin/env python3
from typing import Optional, Dict, Tuple
import os, io, base64, time, json
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import requests
import cv2

"""
LLaVA-1.6 경량 경로(3가지 모드):
  • dummy   : 내부 더미 추론 (바로 테스트용)
  • http    : OpenAI 호환 HTTP 엔드포인트(예: vLLM / LLaVA Serve)  -> 환경변수 LLAVA_API_URL, LLAVA_API_KEY
  • ollama  : Ollama 로컬 서버 (llava:13b 등) -> OLLAMA_HOST 사용

입력:
  /cot/instruction (String)      : 예) "물병 뚜껑을 찾아 열어라" / "노즐을 잡아라"
  /camera/color/image_raw        : RGB/BGR 컬러 이미지
  /camera/aligned_depth_to_color/image_raw : depth (uint16 mm or float32 meters)
  /camera/color/camera_info      : 카메라 내참수

출력:
  /cot/action (String)                     : 예) "move_to_target", "grasp_nozzle", "open_cap"
  /planner/object_3d_refined (PointStamped): 선택된 타깃 3D 좌표(camera frame)
"""

def _depth_to_meters(depth_np: np.ndarray) -> np.ndarray:
    if depth_np.dtype == np.uint16:
        return depth_np.astype(np.float32) * 0.001  # mm -> m
    return depth_np.astype(np.float32)             # 이미 m 라고 가정(float32)

def _project_pixel_to_3d(u:int, v:int, Z:float, K:Tuple[float,float,float,float]):
    fx, fy, cx, cy = K
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z

class LLaVAClient:
    def __init__(self, mode:str="dummy"):
        self.mode = mode
        self.api_url = os.environ.get("LLAVA_API_URL", "")
        self.api_key = os.environ.get("LLAVA_API_KEY", "")
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = os.environ.get("LLAVA_MODEL", "llava-1.6-7b")

    def infer(self, rgb: np.ndarray, instruction:str, context: Optional[Dict]=None) -> Dict:
        H, W = rgb.shape[:2]
        # 1) 더미: 중앙 근처 가우시안 히트맵 + 룰기반 액션
        if self.mode == "dummy":
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            cy, cx = int(H*0.55), int(W*0.55)
            heat = np.exp(-(((yy - cy)**2)/(2*(0.08*H)**2) + ((xx - cx)**2)/(2*(0.08*W)**2)))
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
            if ("노즐" in instruction) or ("nozzle" in instruction.lower()):
                action = "grasp_nozzle"
            elif any(k in instruction for k in ["뚜껑","캡","cap","열어라","open"]):
                action = "open_cap"
            else:
                action = "move_to_target"
            return {"heatmap": heat.astype(np.float32), "action": action}

        # 2) OpenAI 호환 HTTP (vLLM/서드파티 서버)
        if self.mode == "http" and self.api_url:
            # 이미지 base64 인코딩
            ok_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".jpg", ok_rgb)
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role":"system","content":"You are a vision-language assistant for robot actions."},
                    {"role":"user","content":[
                        {"type":"text","text": instruction},
                        {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}
                    ]}
                ],
                "max_tokens": 64,
                "temperature": 0.2,
            }
            headers = {"Content-Type":"application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            try:
                resp = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=10)
                txt = resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                txt = f"move_to_target  # http_error: {e}"
            # 액션 룰 추출(간단히)
            lower = txt.lower()
            if "grasp" in lower or "잡" in lower or "nozzle" in lower:
                action = "grasp_nozzle"
            elif "open" in lower or "열" in lower or "cap" in lower:
                action = "open_cap"
            else:
                action = "move_to_target"
            # 히트맵은 임시 중앙치
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            heat = np.exp(-(((yy - H*0.5)**2)/(2*(0.1*H)**2) + ((xx - W*0.5)**2)/(2*(0.1*W)**2)))
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
            return {"heatmap": heat.astype(np.float32), "action": action}

        # 3) Ollama (멀티모달 플러그인/프롬프팅 방식 다양 → 단순 텍스트 액션만 사용)
        if self.mode == "ollama":
            try:
                # 이미지 전송 없이 텍스트만 사용(간단화)
                resp = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": self.model_name, "prompt": f"[ActionOnly] {instruction}"},
                    timeout=10,
                )
                txt = resp.json().get("response","move_to_target")
            except Exception as e:
                txt = f"move_to_target  # ollama_error: {e}"
            lower = txt.lower()
            if "grasp" in lower or "잡" in lower or "nozzle" in lower:
                action = "grasp_nozzle"
            elif "open" in lower or "열" in lower or "cap" in lower:
                action = "open_cap"
            else:
                action = "move_to_target"
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            heat = np.exp(-(((yy - H*0.5)**2)/(2*(0.1*H)**2) + ((xx - W*0.5)**2)/(2*(0.1*W)**2)))
            heat = (heat - heat.min()) / (heat.max() - 1e-9)
            return {"heatmap": heat.astype(np.float32), "action": action}

        # fallback
        return LLaVAClient(mode="dummy").infer(rgb, instruction, context)

class CoTVLA_LLaVA_Node(Node):
    def __init__(self):
        super().__init__("cot_llava_node")
        self.bridge = CvBridge()
        self.last_rgb: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.K = (600.0, 600.0, 320.0, 240.0)  # fx, fy, cx, cy (기본값)
        self.depth_ok = False
        self.mode = os.environ.get("LLAVA_MODE","dummy")  # dummy|http|ollama
        self.client = LLaVAClient(self.mode)

        # Parameters
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 2.5)
        self.declare_parameter("pixel_suppression", 9)  # 최대픽셀 근방 n×n 억제(옵션)

        self.min_z = float(self.get_parameter("min_depth_m").value)
        self.max_z = float(self.get_parameter("max_depth_m").value)
        self.psup  = int(self.get_parameter("pixel_suppression").value)

        # I/O
        self.sub_ins = self.create_subscription(String, "/cot/instruction", self.on_instruction, 10)
        self.sub_rgb = self.create_subscription(Image, "/camera/color/image_raw", self.on_rgb, 10)
        self.sub_d   = self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.on_depth, 10)
        self.sub_cam = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_caminfo, 10)

        self.pub_action = self.create_publisher(String, "/cot/action", 10)
        self.pub_pt3d   = self.create_publisher(PointStamped, "/planner/object_3d_refined", 10)

        self.get_logger().info(f"🧠 LLaVA wrapper ready (mode={self.mode})")

    def on_caminfo(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx,fy,cx,cy

    def on_rgb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.last_rgb = img

    def on_depth(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg)
        self.last_depth = _depth_to_meters(depth)
        self.depth_ok = True

    def pick_peak(self, heat: np.ndarray) -> Tuple[int,int]:
        # 가장 높은 위치 (nms 대략화)
        yy, xx = np.unravel_index(np.argmax(heat), heat.shape)
        if self.psup > 1:
            r = self.psup//2
            y0, y1 = max(0, yy-r), min(heat.shape[0], yy+r+1)
            x0, x1 = max(0, xx-r), min(heat.shape[1], xx+r+1)
            heat[y0:y1, x0:x1] *= 0.5
        return int(xx), int(yy)  # (u,v)

    def on_instruction(self, msg: String):
        ins = msg.data.strip()
        if self.last_rgb is None:
            self.get_logger().warn("RGB 없음 → 스킵")
            return

        # 1) LLaVA 추론
        out = self.client.infer(self.last_rgb, ins, context=None)
        action = out.get("action", "move_to_target")
        heat   = out.get("heatmap", None)

        # 2) 액션 먼저 publish (로봇 state machine이 듣도록)
        self.pub_action.publish(String(data=action))
        self.get_logger().info(f"🧠 Action: {action}")

        # 3) 히트맵 기반 3D 좌표 산출(있을 때만)
        if heat is None or not self.depth_ok or self.last_depth is None:
            return

        u, v = self.pick_peak(heat)
        Z = float(self.last_depth[v, u])  # meters
        if not np.isfinite(Z) or Z < self.min_z or Z > self.max_z:
            self.get_logger().warn(f"Depth invalid @({u},{v}) Z={Z:.3f}m")
            return

        fx, fy, cx, cy = self.K
        X, Y, Z = _project_pixel_to_3d(u, v, Z, (fx, fy, cx, cy))

        pt = PointStamped()
        pt.header.frame_id = "camera_color_optical_frame"  # 사용 중인 frame으로 맞추세요
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y, pt.point.z = float(X), float(Y), float(Z)
        self.pub_pt3d.publish(pt)
        self.get_logger().info(f"🎯 refined 3D (cam): X={X:.3f} Y={Y:.3f} Z={Z:.3f} @({u},{v})")
        

def main(args=None):
    rclpy.init(args=args)
    node = CoTVLA_LLaVA_Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
