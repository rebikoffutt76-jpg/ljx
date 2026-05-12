#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
np.float = np.float64
import ros_numpy
from sensor_msgs.msg import PointCloud2

class LidarROS:
    def __init__(self):
        rospy.init_node("lidar_test_node")

        rospy.loginfo("正在启动 LiDAR 节点...")
        
        # 订阅你在 rostopic list 中找到的 Ouster 话题
        self.sub = rospy.Subscriber(
            "/ouster/points", 
            PointCloud2, 
            self.pc_callback, 
            queue_size=1,
            buff_size=2**24
        )
        rospy.loginfo("节点启动成功！正在监听 /ouster/points ...")

    def pc_callback(self, msg):
        try:
            # 1. 使用 ros_numpy 将 ROS 点云消息极速转换为 Numpy 结构化数组
            pc_data = ros_numpy.numpify(msg)
            
            # 2. 提取 X, Y, Z 坐标
            # 去除无效的 NaN 点（防止报错）
            mask = np.isfinite(pc_data['x']) & np.isfinite(pc_data['y']) & np.isfinite(pc_data['z'])
            
            # 将离散的 xyz 组合成一个形状为 (N, 3) 的矩阵
            points_3d = np.zeros((np.sum(mask), 3))
            points_3d[:, 0] = pc_data['x'][mask]
            points_3d[:, 1] = pc_data['y'][mask]
            points_3d[:, 2] = pc_data['z'][mask]

            # 3. 数据分析小测试：计算最近的障碍物
            num_points = points_3d.shape[0]
            
            if num_points > 0:
                # 计算每个点到雷达原点 (0,0,0) 的直线距离： 根号下(x^2 + y^2 + z^2)
                distances = np.linalg.norm(points_3d, axis=1)
                
                # 过滤掉距离为 0 的点（通常是雷达自身）和距离太远的点
                valid_distances = distances[(distances > 0.2) & (distances < 50.0)]
                
                if len(valid_distances) > 0:
                    closest_dist = np.min(valid_distances)
                    rospy.loginfo(f"接收到 {num_points} 个点! 最近的障碍物距离机器人: {closest_dist:.2f} 米")
                else:
                    rospy.loginfo(f"接收到 {num_points} 个点! 附近没有明显障碍物。")

        except Exception as e:
            rospy.logerr(f"处理点云时发生错误: {e}")

if __name__ == "__main__":
    try:
        LidarROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
