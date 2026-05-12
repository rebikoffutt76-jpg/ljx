#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class YoloROS:
    def __init__(self):
        rospy.init_node("yolo_ros_node")

        rospy.loginfo("正在加载 YOLOv8 模型...")
        self.model = YOLO("yolov8n.pt") 
        rospy.loginfo("模型加载成功！等待接收图像...")

 
        self.bridge = CvBridge()


        self.sub = rospy.Subscriber(
            "/rgb/image_raw",
            Image, 
            self.image_callback, 
            queue_size=1, 
            buff_size=2**24
        )

        self.pub_img = rospy.Publisher("/yolo/result_image", Image, queue_size=1)

    def image_callback(self, msg):
        try:

            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            results = self.model(frame, verbose=False)
            boxes = results[0].boxes 

            if boxes is not None and len(boxes) > 0:
                rospy.loginfo("======= 当前画面检测结果 =======") 
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    label_name = self.model.names[cls] 
                    label_text = f"{label_name} {conf:.2f}"
                    
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    rospy.loginfo(f"发现目标: [{label_name}], 置信度: {conf:.2f}, 坐标: X={center_x:.1f}, Y={center_y:.1f}")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 将画好框的图像安全转回 ROS 格式并发布
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.pub_img.publish(img_msg)

        except CvBridgeError as e:
            rospy.logerr(f"图像转换失败: {e}")
        except Exception as e:
            rospy.logerr(f"处理时发生未知错误: {e}")

if __name__ == "__main__":
    try:
        YoloROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
