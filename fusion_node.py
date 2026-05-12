#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
np.float = np.float64  # 修复 numpy 版本冲突

import ros_numpy
from sensor_msgs.msg import Image, PointCloud2
from ultralytics import YOLO
from cv_bridge import CvBridge

class SensorFusionNode:
    def __init__(self):
        rospy.init_node("sensor_fusion_node")

        # 1. 加载 YOLO
        rospy.loginfo("加载 YOLO 大脑...")
        self.model = YOLO("yolov8n.pt")
        self.bridge = CvBridge()

        # 2. 你的专属相机内参 (来自 rostopic echo)
        self.fx = 609.15686
        self.fy = 609.10510
        self.cx = 641.17797
        self.cy = 364.35092

        # 3. 存储最新的雷达点云数据
        self.latest_pc = None

        # 4. 订阅雷达 (保存最新一帧数据)
        rospy.Subscriber("/ouster/points", PointCloud2, self.pc_callback, queue_size=1)
        
        # 5. 订阅相机 (触发推理和融合计算)
        rospy.Subscriber("/rgb/image_raw", Image, self.image_callback, queue_size=1)
        
        # 6. 发布融合后的图像
        self.pub_img = rospy.Publisher("/yolo/fusion_image", Image, queue_size=1)
        
        rospy.loginfo("视觉雷达融合节点启动成功,等待数据交汇...")

    def pc_callback(self, msg):
        try:
            pc_data = ros_numpy.numpify(msg)
            mask = np.isfinite(pc_data['x']) & np.isfinite(pc_data['y']) & np.isfinite(pc_data['z'])
            
            # 获取雷达坐标系下的 XYZ
            x_lidar = pc_data['x'][mask]
            y_lidar = pc_data['y'][mask]
            z_lidar = pc_data['z'][mask]

            # 【坐标系转换】：把雷达坐标系转成相机坐标系
            # 通常雷达(X前, Y左, Z上) -> 相机(Z前, X右, Y下)
            X_cam = -y_lidar
            Y_cam = -z_lidar
            Z_cam = x_lidar
            
            # 只保留在相机正前方的点 (Z > 0)
            front_mask = Z_cam > 0.1
            self.latest_pc = np.vstack((X_cam[front_mask], Y_cam[front_mask], Z_cam[front_mask])).T
            
        except Exception as e:
            pass # 忽略偶尔的丢包

    def image_callback(self, msg):
        if self.latest_pc is None:
            return # 如果还没收到雷达数据，先不处理图像

        try:
            # 获取图像
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 获取点云副本，防止被篡改
            points_3d = self.latest_pc.copy()

            # 【核心公式】：把所有 3D 点投影到 2D 像素坐标 (u, v)
            u = (self.fx * points_3d[:, 0] / points_3d[:, 2]) + self.cx
            v = (self.fy * points_3d[:, 1] / points_3d[:, 2]) + self.cy
            
            # YOLO 目标检测
            results = self.model(frame, verbose=False)
            boxes = results[0].boxes 

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label_name = self.model.names[cls] 
                    
                    # 找出哪些雷达点落在了这个物体的 2D 绿框
                    inside_box_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                    points_in_box = points_3d[inside_box_mask]

                    distance_str = "Unknown"
                    if len(points_in_box) > 10: # 如果框里至少有 10 个有效雷达点
                        # 取距离中位数作为目标的真实距离（防噪点）
                        true_distance = np.median(points_in_box[:, 2])
                        distance_str = f"{true_distance:.2f} m"
                        rospy.loginfo(f"锁定目标: [{label_name}], 真实距离: {true_distance:.2f} 米")
                    
                    # 画图：在框上标出距离！
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label_text = f"{label_name} | {distance_str}"
                    cv2.putText(frame, label_text, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 发布最终画面
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.pub_img.publish(img_msg)

        except Exception as e:
            rospy.logerr(f"融合错误: {e}")

if __name__ == "__main__":
    try:
        SensorFusionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
