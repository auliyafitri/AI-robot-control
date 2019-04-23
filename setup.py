import sys
import os.path
from setuptools import setup, find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AI-robot-control'))

setup(name='AI-robot-control',
      version='0.0.1',
      packages=[package for package in find_packages()
                if package.startswith('AI-robot-control')],
      description='AI-robot-control: Developing and comparing reinforcement learning agents using Gazebo, ROS, and Baselines',
      url='https://github.com/PhilipKurrek/AI-robot-control',
      author='Logistics Robotics',
)
