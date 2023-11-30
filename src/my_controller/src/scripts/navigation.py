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

        grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # trainimage

        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)

        matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)

        good_points = []
        
        
        ### Contours garbage

         
        # Find contours in the binary mask
        #contours, _ = cv2.findContours(blue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        c_img = cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
       
        if contours:
            # Find the largest rectangular contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the image to the bounding rectangle
            cropped_img = c_img[y:y+h, x:x+w]
            cv2.imshow("cropped image", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)    

        pts1 = np.float32([[y,x],[y,x+w],[y+h,x],[y+h,x+w]])
        pts2 = np.float32([[0,0],[0,600],[400,0],[400,600]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        #perspective = frame.copy()
        dst = cv2.warpPerspective(cropped_img,M,(600,400))


        cv2.imshow("contours", dst)
        cv2.waitKey(1)
        
        ###
        
        for m, n in matches:
            if m.distance < 0.25 * n.distance:
                good_points.append(m)

        if len(good_points) > 4:
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            matrix, _ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

            # Perspective transform
            h, w = self.img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            cv2.imshow("Image window", cv2.cvtColor(homography, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            cv2.imshow("Binary", blue_mask)
            cv2.waitKey(1)
            
            


        else:
            cv2.imshow("Image window", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            cv2.imshow("Binary", blue_mask)
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