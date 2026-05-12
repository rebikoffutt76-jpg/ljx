#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

# 兼容性处理：解决部分环境下 numpy 与 ros_numpy 的类型定义冲突
np.float = np.float64  

import ros_numpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, PointCloud2
from ultralytics import YOLO
from cv_bridge import CvBridge

class ScoutFollower:
    def __init__(self):
        rospy.init_node("scout_follower_node")
        
        # 1. 控制系统参数配置
        # 线速度控制参数 (基于距离误差的比例控制)
        self.target_dist = 1.5      # 目标保持距离 (单位: 米)
        self.linear_k = 0.4         # 线性增益系数
        
        # 角速度控制参数 (基于像素误差的 PD 控制)
        self.angular_p = 0.0025     # 比例增益
        self.angular_d = 0.008      # 微分增益 (阻尼)
        self.deadzone_x = 60        # 航向角控制死区 (单位: 像素)
        self.img_center_x = 640     # 图像水平中心参考值 (1280px / 2)
        self.last_error_x = 0.0     # 历史误差项，用于微分计算

        # 2. 状态机与安全参数
        self.last_valid_cmd = Twist()
        self.last_seen_time = rospy.Time.now()
        self.retention_time = 0.5   # 目标丢失后的状态保持时限 (单位: 秒)
        
        self.robot_width = 0.6      # 机器人模型宽度，用于避障区域定义
        self.danger_z = 0.2         # 虚拟保险杠纵向触发阈值 (单位: 米)

        # 3. 相机内参配置
        self.fx = 609.15686
        self.fy = 609.10510
        self.cx = 641.17797
        self.cy = 364.35092

        # 4. 算法组件初始化
        rospy.loginfo("Initializing YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")
        self.bridge = CvBridge()
        self.latest_pc = None
        
        # 5. ROS 通信接口定义
        # 订阅激光雷达点云与相机原始图像
        rospy.Subscriber("/ouster/points", PointCloud2, self.pc_callback, queue_size=1)
        rospy.Subscriber("/rgb/image_raw", Image, self.image_callback, queue_size=1, tcp_nodelay=True)
        
        # 发布底盘控制指令与调试图像流
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1) 
        self.debug_pub = rospy.Publisher("/follower/debug_image", Image, queue_size=1)
        
        rospy.loginfo("System initialized. Monitoring /follower/debug_image for visualization.")

    def pc_callback(self, msg):
        """
        处理激光雷达点云数据，执行坐标系转换与预滤波
        """
        try:
            pc_data = ros_numpy.numpify(msg)
            # 过滤非数值点
            mask = np.isfinite(pc_data['x']) & np.isfinite(pc_data['y']) & np.isfinite(pc_data['z'])
            x_lidar, y_lidar, z_lidar = pc_data['x'][mask], pc_data['y'][mask], pc_data['z'][mask]

            # 坐标系变换：LiDAR 坐标系 -> 摄像头光学坐标系
            # 变换关系取决于传感器的物理安装位姿，当前配置为标准正向安装
            X_cam, Y_cam, Z_cam = -y_lidar, -z_lidar, x_lidar
            
            # 仅保留前方半空间有效点
            front_mask = Z_cam > 0.1
            self.latest_pc = np.vstack((X_cam[front_mask], Y_cam[front_mask], Z_cam[front_mask])).T
        except Exception as e:
            rospy.logerr(f"LiDAR data processing error: {e}")

    def check_virtual_bumper(self):
        """
        基于点云空间分布的避障检测逻辑
        """
        if self.latest_pc is None: 
            return False
        
        # 定义机器人正前方的矩形占用区域
        danger_zone_mask = (self.latest_pc[:, 2] < self.danger_z) & \
                           (self.latest_pc[:, 2] > 0.1) & \
                           (np.abs(self.latest_pc[:, 0]) < (self.robot_width / 2.0))
                           
        # 统计异常障碍点密度，滤除散乱噪声
        if np.sum(danger_zone_mask) > 40: 
            return True
        return False

    def image_callback(self, msg):
        """
        主控制循环：感知融合、决策生成与指令发布
        """
        try:
            # 图像解码
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            current_time = rospy.Time.now()

            # 系统状态自检：传感器数据同步检查
            if self.latest_pc is None:
                rospy.logwarn_throttle(5, "Waiting for LiDAR point cloud data...")
                cv2.putText(frame, "STATUS: WAITING FOR LiDAR", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
                return 
                
            # 系统状态自检：处理延时检查
            delay = (current_time - msg.header.stamp).to_sec()
            if delay > 0.15:
                rospy.logwarn_throttle(2, f"High latency detected: {delay:.2f}s. Skipping frame.")
                return 

            points_3d = self.latest_pc.copy()
            
            # 行为优先级 1：紧急避障响应
            if self.check_virtual_bumper():
                rospy.logwarn_throttle(1, "Obstacle detected. Triggering emergency avoidance.")
                cv2.putText(frame, "STATE: EMERGENCY AVOIDANCE", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                emergency_cmd = Twist()
                emergency_cmd.linear.x = 0  # 逆向运动
                emergency_cmd.angular.z = 0   # 偏航脱离
                self.cmd_pub.publish(emergency_cmd)
                self.last_valid_cmd = emergency_cmd
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
                return 

            # 空间投影计算：3D 空间点映射至 2D 像素平面
            u = (self.fx * points_3d[:, 0] / points_3d[:, 2]) + self.cx
            v = (self.fy * points_3d[:, 1] / points_3d[:, 2]) + self.cy

            # 语义感知：YOLOv8 推理 (采用 320px 缩放以优化实时性能)
            results = self.model(frame, imgsz=320, verbose=False)
            boxes = results[0].boxes
            
            twist = Twist() 
            target_detected = False

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    # 仅处理 'person' 类别
                    if self.model.names[cls] == 'person':
                        conf = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        center_x = (x1 + x2) / 2.0
                        
                        # 空间融合：提取边界框内的 3D 深度信息
                        inside_box_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                        points_in_box = points_3d[inside_box_mask]

                        if len(points_in_box) > 10:
                            # 鲁棒距离估计：采用深度中位数滤波
                            dist = np.median(points_in_box[:, 2])
                            
                            # 决策生成：PD 航向控制
                            error_x = self.img_center_x - center_x
                            if abs(error_x) < self.deadzone_x:
                                twist.angular.z = 0.0
                                self.last_error_x = 0.0 
                            else:
                                error_diff = error_x - self.last_error_x
                                twist.angular.z = (error_x * self.angular_p) + (error_diff * self.angular_d)
                                self.last_error_x = error_x

                            # 决策生成：线性距离控制
                            twist.linear.x = (dist - self.target_dist) * self.linear_k
                            
                            # 底盘动力学约束限制
                            twist.linear.x = np.clip(twist.linear.x, -0.3, 0.3)
                            twist.angular.z = np.clip(twist.angular.z, -0.5, 0.5)

                            target_detected = True
                            self.last_seen_time = current_time
                            self.last_valid_cmd = twist
                            
                            # 可视化增强
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Target: {dist:.2f}m (Conf: {conf:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break 

            # 行为优先级 2：状态保持机制 (针对感知瞬时失效的鲁棒性优化)
            if not target_detected:
                time_since_last_seen = (current_time - self.last_seen_time).to_sec()
                if time_since_last_seen < self.retention_time:
                    twist = self.last_valid_cmd 
                    cv2.putText(frame, "STATE: TRACKING (ESTIMATED)", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    # 状态重置
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.last_error_x = 0.0 
                    self.last_valid_cmd = twist
                    cv2.putText(frame, "STATE: TARGET LOST", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 指令发布
            self.cmd_pub.publish(twist)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Main loop execution error: {e}")

if __name__ == "__main__":
    try:
        node = ScoutFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
