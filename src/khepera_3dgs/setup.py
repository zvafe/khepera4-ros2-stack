import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'khepera_3dgs'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andrei Berteanu',
    maintainer_email='student@academic.edu',
    description='Decoupled, drift-free 3D Gaussian Splatting reconstruction pipeline for Khepera IV mobile robot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pipeline_node = khepera_3dgs.pipeline_node:main',
            'train_khepera_3dgs = khepera_3dgs.scripts_runner:main_cli',
        ],
    },
)
