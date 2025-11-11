from setuptools import setup

package_name = 'cot_vision_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/vision_cot_bringup.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dongfan',
    maintainer_email='df@example.com',
    description='CoT-VLA Vision + Speech integration for SmartFuel Robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_cot_node = cot_vision_ros2.vision_cot_node:main',
            'cot_llava_node = cot_vision_ros2.cot_llava_node:main',
            'cot_action_executor = cot_vision_ros2.cot_action_executor:main',
            'license_plate_reader = cot_vision_ros2.license_plate_reader:main',
            'speech_feedback = cot_vision_ros2.speech_feedback:main',
        ],
    },
)
