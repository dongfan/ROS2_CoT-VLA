#!/usr/bin/env python3
import cv2, pytesseract, os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from gtts import gTTS

class LicensePlateReader(Node):
    def __init__(self):
        super().__init__('license_plate_reader')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/color/image_raw', self.on_image, 10)
        self.pub_plate = self.create_publisher(String, '/fuel/license_plate', 10)
        self.said = False

    def speak(self, text):
        tts = gTTS(text=text, lang='ko')
        path = '/tmp/tts.mp3'
        tts.save(path)
        os.system(f"mpg123 -q {path}")

    def on_image(self, msg):
        if self.said:  # 한 번만 실행
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        text = pytesseract.image_to_string(gray, lang='kor', config='--psm 7')
        text = text.strip().replace(' ', '')
        if len(text) > 5:
            msg_str = f"번호판 {text} 확인되었습니다."
            self.pub_plate.publish(String(data=msg_str))
            self.get_logger().info(msg_str)
            self.speak(msg_str)
            self.said = True

def main():
    rclpy.init()
    node = LicensePlateReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
