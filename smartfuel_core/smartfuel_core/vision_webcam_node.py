#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import glob
import json
from ultralytics import YOLO

class VisionWebcamNode(Node):
    def __init__(self):
        super().__init__('vision_webcam_node')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # YOLOv8 기본 모델 사용
        self.pub_image = self.create_publisher(Image, '/fuel/webcam_color', 10)
        self.pub_result = self.create_publisher(String, '/fuel/image_result', 10)
        self.pub_detected = self.create_publisher(String, '/car_detected', 10)

        # 카메라 장치 선택
        device_index = "/dev/v4l/by-id/usb-Generic_HD_camera-video-index0"
        if not glob.glob(device_index):
            devices = glob.glob('/dev/video*')
            if not devices:
                raise RuntimeError("❌ No /dev/video* devices found.")
            device_index = devices[0]

        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Webcam not opened: {device_index}")

        self.get_logger().info(f"✅ Webcam opened: {device_index}")
        self.timer = self.create_timer(0.05, self.loop)

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ Webcam frame not received")
            return

        # YOLO 추론
        results = self.model(frame, verbose=False)
        detections = []
        car_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls]
                if label in ['car', 'truck', 'bus'] and conf > 0.5:
                    car_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                detections.append({"label": label, "conf": conf})

        # Publish results
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub_image.publish(img_msg)
        self.pub_result.publish(String(data=json.dumps(detections)))
        if car_detected:
            self.pub_detected.publish(String(data="detected"))
            self.get_logger().info("🚗 차량 감지됨")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("📷 Webcam released")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionWebcamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 VisionWebcamNode 종료됨 (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
