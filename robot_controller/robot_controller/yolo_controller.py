import numpy as np
import math
import cv2

import rclpy
from rclpy.node import Node

from custom_msg.msg import BoundingBox, BoundingBoxes
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image

from cv_bridge import CvBridge, CvBridgeError


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        
        self.sub_bbxes = self.create_subscription(BoundingBoxes, '/bounding_box_array', self.DetectionCallback, 10)
        self.sub_img = self.create_subscription(Image, '/turtlebot/camera/image_raw', self.ImageCallback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/turtlebot/scan', self.ScanCallback, 10)
        self.pub_vel = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)
        
        self.cmd_vel = Twist()
        self.ranges = LaserScan()
        self.cv_img = np.zeros((640, 480, 3), np.uint8)
        
        self.hist_ref = 0.0
        
        self.turn_ref = 320.0
        self.prev_turn_err = 0.0
        
        self.dist_ref = 2.5
        self.prev_dist_err = 0.0
        
        self.ang_limit = 0.4
        self.lin_limit = 0.5
        
        self.mode = 0

    def DetectionCallback(self, data):
        if self.mode < 100:
            # Set following people
            # when first detection
            bbx_xmin = data.bounding_boxes[0].xmin
            bbx_ymin = data.bounding_boxes[0].ymin
            bbx_xmax = data.bounding_boxes[0].xmax
            bbx_ymax = data.bounding_boxes[0].ymax
            
            obj_img = self.cv_img[bbx_ymin:bbx_ymax, bbx_xmin:bbx_xmax]
            
            hist,bins = np.histogram(obj_img.ravel(), 256, [0,256])
            self.hist_ref = np.std(hist)
            print(self.hist_ref)
            
            self.mode += 1
            
            cv2.imshow('Interesting obj', obj_img)
            cv2.waitKey(1)
        else:
            obj_xmin = 0.0
            obj_xmax = 0.0
            obj_img = np.zeros((640, 480, 3), np.uint8)
            hist_err_min = 9999.0
            for bbx in data.bounding_boxes:
                # Find interesting obj between objs
                bbx_xmin = bbx.xmin
                bbx_ymin = bbx.ymin
                bbx_xmax = bbx.xmax
                bbx_ymax = bbx.ymax
                
                obj_img = self.cv_img[bbx_ymin:bbx_ymax, bbx_xmin:bbx_xmax]
                hist,bins = np.histogram(obj_img.ravel(), 256, [0,256])
                hist = np.std(hist)
                
                if abs(hist - self.hist_ref) < hist_err_min:
                    obj_xmin = bbx_xmin
                    obj_xmax = bbx_xmax
                
            cv2.imshow('Interesting obj', obj_img)
            cv2.waitKey(1)
            
            # Turn
            obj_x = (obj_xmin + obj_xmax) / 2
            
            kp = 0.001
            kd = 0.0008
            turn_err = self.turn_ref - obj_x
            turn = kp * turn_err + kd * (turn_err - self.prev_turn_err)
            if turn > self.ang_limit:           # Angular limit
                turn = self.ang_limit
            elif turn < -self.ang_limit:        # Angular limit
                turn = -self.ang_limit
            self.prev_turn_err = turn_err
            
            # Follow people
            speed = 0.0
            if abs(turn) < 0.01:
                # Front ranges avg
                # front_dist = np.sum(self.ranges.ranges[-5:]) + np.sum(self.ranges.ranges[:5])
                front_dist = self.ranges.ranges[-1] + self.ranges.ranges[1]
                front_dist = front_dist / 2
                
                kp2 = 0.1
                kd2 = 0.08
                dist_err = front_dist - self.dist_ref
                speed = kp2 * dist_err + kd2 * (dist_err - self.prev_dist_err)
                if speed > self.lin_limit:      # Linear limit
                    speed = self.lin_limit
                elif speed < -self.lin_limit:   # Linear limit
                    speed = -self.lin_limit
                elif math.isnan(speed):
                    speed = 0.2
                self.prev_dist_err = dist_err
            
            self.cmd_vel.linear.x = speed
            self.cmd_vel.angular.z = turn
            self.pub_vel.publish(self.cmd_vel)
            print('speed: ', speed, 'turn: ', turn)
        
    def ImageCallback(self, data):
        bridge = CvBridge()
        try:
            cv_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            cv_img = cv2.resize(cv_img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            self.cv_img = cv_img
        except CvBridgeError as e:
            print(e)
        
    def ScanCallback(self, data):
        self.ranges = data


def main(args=None):
    rclpy.init(args=args)

    n = ControllerNode()

    rclpy.spin(n)

    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    