import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'khepera4_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andy',
    maintainer_email='andy@todo.todo',
    description='Closed-loop and open-loop control packages for Khepera IV mobile robot',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'open_loop_benchmark = khepera4_control.open_loop_benchmark:main',
            'closed_loop_controller = khepera4_control.closed_loop_controller:main',
            'telemetry_logger = khepera4_control.telemetry_logger:main',
            'control_dashboard = khepera4_control.dashboard:main',
        ],
    },
)
