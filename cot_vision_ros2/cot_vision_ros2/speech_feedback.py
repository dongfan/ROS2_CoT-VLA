#!/usr/bin/env python3
import rclpy, os, time
from rclpy.node import Node
from std_msgs.msg import String
from gtts import gTTS

class SpeechFeedback(Node):
    def __init__(self):
        super().__init__('speech_feedback')
        self.sub_plate = self.create_subscription(String, '/fuel/license_plate', self.on_plate, 10)
        self.sub_action = self.create_subscription(String, '/cot/action', self.on_action, 10)
        self.sub_cap = self.create_subscription(String, '/fuel/cap_opened', self.on_cap_opened, 10)
        self.state = 'idle'
        self.timer = None
        self.get_logger().info("🔊 Speech feedback FSM started")

    def speak(self, text):
        tts = gTTS(text=text, lang='ko')
        path = '/tmp/tts.mp3'
        tts.save(path)
        os.system(f"mpg123 -q {path}")

    def on_plate(self, msg):
        self.state = 'plate_confirmed'
        self.speak(msg.data)
        time.sleep(2)
        self.speak("주유를 시작하겠습니다.")
        self.state = 'orienting'
        self.publish_action("orient_minus_y")

        time.sleep(3)
        self.speak("주유구를 찾는 중입니다.")
        self.state = 'searching'

    def on_action(self, msg):
        action = msg.data.lower()
        if self.state == 'searching' and ('move_to_target' in action or 'open_cap' in action):
            self.speak("주유구를 찾았습니다.")
            self.speak("주유구를 열어주세요.")
            self.state = 'waiting_open'

    def on_cap_opened(self, msg):
        if self.state == 'waiting_open':
            self.speak("주유구를 열겠습니다.")
            self.publish_action("open_cap")
            self.state = 'opening'

    def publish_action(self, action_name: str):
        pub = self.create_publisher(String, '/cot/action', 10)
        pub.publish(String(data=action_name))
        self.get_logger().info(f"📤 Action triggered → {action_name}")

def main():
    rclpy.init()
    node = SpeechFeedback()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
