from dataclasses import dataclass
import numpy as np

@dataclass
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_camera_info(msg) -> "PinholeIntrinsics":
        k = msg.k  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        return PinholeIntrinsics(fx=k[0], fy=k[4], cx=k[2], cy=k[5])


def deproject_pixel_to_camera(u: float, v: float, depth_m: float, K: PinholeIntrinsics):
    # depth in meters, returns (Xc, Yc, Zc) in camera frame
    if depth_m <= 0 or np.isnan(depth_m) or np.isinf(depth_m):
        return None
    X = (u - K.cx) * depth_m / K.fx
    Y = (v - K.cy) * depth_m / K.fy
    Z = depth_m
    return (X, Y, Z)
