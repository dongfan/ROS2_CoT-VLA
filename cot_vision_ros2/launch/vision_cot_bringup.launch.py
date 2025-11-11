from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cot_vision_ros2',
            executable='vision_cot_node',
            name='vision_cot',
            output='screen',
            parameters=[{
                'depth_scale': 0.001,
                'depth_window': 7,
                'heat_thresh': 0.6,
                'min_depth_m': 0.1,
                'max_depth_m': 4.0,
            }]
        ),
    ])
