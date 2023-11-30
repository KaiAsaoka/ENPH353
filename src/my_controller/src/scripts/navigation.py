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
from geometry_msgs.msg import Twist
import csv


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
        self.move_pub = rospy.Publisher("/R1/cmd_vel",Twist,queue_size=1)
        print("Loaded template image file: " + self.template_path)

        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
  

    def callback(self, data):

        WIDTH = 600
        HEIGHT = 400
        
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
        # Apply blue color mask

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([90, 50, 30])
        upper_blue = np.array([120, 255, 120])
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        if contours:

            largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        contour = frame.copy()
        cv2.drawContours(contour, [approx], 0, (0, 255, 0), 2)

        tl, tr, bl, br = rectangle_positions(approx)
            
        pts1 = np.float32([tl,tr,bl,br])
        pts2 = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])

        perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        dst = cv2.warpPerspective(frame, perspective_matrix, (WIDTH, HEIGHT))

        # Display 
        cv2.imshow("Contour Crop", cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        cv2.imshow("Image window", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        cv2.imshow("Binary", blue_mask)
        cv2.waitKey(1)

        cv2.imshow("Contour Image", cv2.cvtColor(contour, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        csv_file_path = '/home/fizzer/ros_ws/src/2023_competition/enph353/enph353_gazebo/scripts/plates.csv'

        # Open the CSV file and read its contents
        with open(csv_file_path, 'r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)

            # Read the data from the CSV file
            for row in csv_reader:
                print(row)

    ##
    # @brief Listener method to retrieve data from ROS camera
        ##NAVIGATION 

        ## Define the coordinates of the region of interest (ROI)
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 400, 1280, 410  # Adjust these coordinates as needed
        ## Default Resolution x = 320, y = 240

        ## Crop the image to the ROI
        roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        cv2.waitKey(1)
        
        hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        ## Define the lower and upper bounds for the color you want to detect (here, it's blue)
        sensitivity = 15
        cv2.waitKey(1)
        ## Define a threshold value for detecting grayscale change
        threshold_value = 100  # Adjust this threshold as needed

        ## Find contours in the binary mask
        pidcontours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx = 0
        cx = 0

        pid_img = cv2.drawContours(frame, pidcontours, -1, (0, 255, 0), 1)

        ## Iterate through the contours and find the position of color change within the ROI
        for contour in pidcontours:

            ## Calculate the centroid of the contour
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                ## Add the ROI offset to get the position within the original image
                
                cx += roi_x1
                cy += roi_y1

                #print(f"Position of color change within ROI: ({cx}, {cy})")

        rate = rospy.Rate(2)
        move = Twist()
        move.linear.x = .1

        if(cx<640):
            move.angular.z = .5
        else:
            move.angular.z = -.5

        cv2.imshow("PID", cv2.cvtColor(pid_img, cv2.COLOR_RGB2BGR))
        self.move_pub.publish(move)
        
        
def rectangle_positions(approx):
    x0, y0 = approx[0][0][0], approx[0][0][1]
    x2, y2 = approx[2][0][0], approx[2][0][1]
    if(x0 < x2):
        return approx[0],approx[3],approx[1],approx[2]
    else:
        return approx[1],approx[0],approx[2],approx[3]

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