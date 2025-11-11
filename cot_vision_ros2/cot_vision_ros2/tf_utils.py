from geometry_msgs.msg import PointStamped
from builtin_interfaces.msg import Time
from typing import Optional, Tuple

class TFBufferProxy:
    def __init__(self, tf_buffer):
        self._buf = tf_buffer

    def transform_point(self, p_cam: Tuple[float, float, float], stamp: Time, from_frame: str, to_frame: str) -> Optional[PointStamped]:
        ps = PointStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = from_frame
        ps.point.x, ps.point.y, ps.point.z = p_cam
        try:
            out = self._buf.transform(ps, to_frame, timeout=rclpy.duration.Duration(seconds=0.1))
            return out
        except Exception:
            return None

