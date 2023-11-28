#! /usr/bin/env python3

##
# @package tapefollow
# @brief A ROS program which enables a robot to follow a line using CV2. 

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import sys


##
# @brief Callback method
# @retval 
class navigation():
    
    
    def __init__(self):
        
        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        self.sift = cv2.SIFT_create()

		## Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.template_path = "/home/fizzer/ros_ws/src/2023_competition/media_src/clue_banner.png"
        self.img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		## Features
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)
        print("Loaded template image file: " + self.template_path)
        
    def callback(self, data):

        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage

        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)

        matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        if len(good_points) > 7:
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            matrix, _ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

            # Perspective transform
            h, w = self.img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            cv2.imshow("Image window", homography)
            cv2.waitKey(1)

        elif len(good_points) > 2:
            matches = cv2.drawMatches(self.img, self.kp_image, frame, kp_grayframe, good_points, None, flags=0)

            cv2.imshow("Image window", matches)
            cv2.waitKey(1)

        else:
            cv2.imshow("Image window", frame)
            cv2.waitKey(1)


    ##
    # @brief Listener method to retrieve data from ROS camera
        
    
def main(args):
    rospy.init_node('listener', anonymous=True)
    controller = navigation()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)