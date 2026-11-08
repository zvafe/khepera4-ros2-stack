import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'khepera_hybrid_nav'

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
    maintainer_email='student@academic.edu',
    description='Master coordinator for hybrid OpenFrontier and SplatSearch navigation.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hybrid_coordinator_node = khepera_hybrid_nav.hybrid_coordinator_node:main'
        ],
    },
)
