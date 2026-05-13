#!/usr/bin/env bash
set -e

# 定义环境变量配置命令 
ROS_SETUP="source /opt/ros/noetic/setup.bash"
WS_NAV="source /home/tony/MFLB-vineyard-navigation/devel/setup.bash"
WS_YOLO="source /home/tony/yltest/devel/setup.bash"

# 1. 启动 Scout 底盘核心
echo ">>> 1. 正在启动 Scout 底盘 (scout_robot_base.launch)..."
gnome-terminal -- bash -c "$ROS_SETUP && $WS_NAV && cd /home/tony/MFLB-vineyard-navigation/src/agilex/scout_ros/scout_bringup/launch && roslaunch scout_robot_base.launch; exec bash"
sleep 4

# 2. 启动传感器系统 (LiDAR + Camera)
echo ">>> 2. 正在启动感知系统 雷达+相机 (bringup_v2.launch)..."
gnome-terminal -- bash -c "$ROS_SETUP && $WS_NAV && cd /home/tony/MFLB-vineyard-navigation/src/topgear_ros/topgear_bringup/launch && roslaunch bringup_v2.launch; exec bash"
sleep 4

# 3. 启动传统导航 (已禁用)
# echo ">>> (跳过) 传统 2D 导航已禁用，防止与跟随冲突"
# gnome-terminal -- bash -c "$ROS_SETUP && $WS_NAV && cd /home/tony/MFLB-vineyard-navigation/src/navigation/launch && roslaunch nav_real.launch; exec bash"
# sleep 3

# 4. 启动 视觉跟随+避障 终极大节点
echo ">>> 3. 正在启动 AI 跟随大脑 (follower_node.py)..."
gnome-terminal -- bash -c "$ROS_SETUP && $WS_YOLO && rosrun yolo_ros_detector follower_node.py; exec bash"
sleep 2

# 5. 自动启动监控面板
echo ">>> 4. 正在打开图像监视器..."
gnome-terminal -- bash -c "$ROS_SETUP && rqt_image_view; exec bash"

