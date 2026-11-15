# Copyright 2026 Andrei Berteanu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package setup configuration for khepera_visual_pursuit."""

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'khepera_visual_pursuit'

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
    description='Hybrid visual pursuit and tracking architecture for Khepera IV mobile robot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detector_verifier_node = '
            'khepera_visual_pursuit.object_detector_verifier_node:main',
            'lightglue_tracker_node = '
            'khepera_visual_pursuit.lightglue_tracker_node:main',
            'ibvs_controller_node = '
            'khepera_visual_pursuit.ibvs_controller_node:main',
        ],
    },
)
