#! /usr/bin/env python3

##
# @package tapefollow
# @brief A ROS program which enables a robot to follow a line using CV2. 

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys

##
# @brief Callback method
# @retval 
class navigation():


    def __init__(self):
        
        
        self.sift = cv2.SIFT_create()

		## Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.template_path = "/home/fizzer/ros_ws/src/2023_competition/media_src/clue_banner.png"
        self.img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		## Features
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)
        self.bridge = CvBridge()
        print("Loaded template image file: " + self.template_path)

        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
  

    def callback(self, data):

        
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([90, 50, 30])
        upper_blue = np.array([120, 255, 120])
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        if contours:

            largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        contour = frame.copy()
        cv2.drawContours(contour, [approx], 0, (0, 255, 0), 2)

        x0, y0 = approx[0][0][0], approx[0][0][1]
        x2, y2 = approx[2][0][0], approx[2][0][1]

        if(x0 < x2):
            tl = approx[0]
            tr = approx[3]
            bl = approx[1]  
            br = approx[2]
        else:
            tl = approx[1]
            tr = approx[0]
            bl = approx[2]  
            br = approx[3]
            
        pts1 = np.float32([tl,tr,bl,br])
        pts2 = np.float32([[0,0],[600,0],[0,400],[600,400]])

        perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        # Apply the perspective transform
        dst = cv2.warpPerspective(frame, perspective_matrix, (600, 400))

        cv2.imshow("Contour Crop", cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        cv2.imshow("Image window", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        cv2.imshow("Binary", blue_mask)
        cv2.waitKey(1)

        cv2.imshow("Contour Image", cv2.cvtColor(contour, cv2.COLOR_RGB2BGR))
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
    finally: cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)