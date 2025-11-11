from setuptools import setup

package_name = 'smartfuel_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools', 'doosan-robot2'],
    zip_safe=True,
    maintainer='df',
    maintainer_email='your@email.com',
    description='SmartFuel Core package: motion + gripper + FSM + vision + server',
    license='Apache License 2.0',
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/smartfuel_core']),
        ('share/smartfuel_core', ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'motion_controller = smartfuel_core.motion_controller:main',
            'vision_webcam_node = smartfuel_core.vision_webcam_node:main',
            'server = smartfuel_core.server:main',
        ],
    },
)
