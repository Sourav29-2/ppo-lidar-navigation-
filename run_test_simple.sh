#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
export ROS_LOCALHOST_ONLY=1
source /Users/souravkumar/ros2_study/install/setup.bash
python3 -c "print('hello from python', flush=True)"
