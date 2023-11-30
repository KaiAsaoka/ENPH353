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

        
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([115, 128, 95])
        upper_blue = np.array([120, 255, 204])
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
        dstmask = cv2.warpPerspective(blue_mask, perspective_matrix, (600, 400))
        letters, _ = cv2.findContours(dstmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        lettermask = dst.copy()
        letterimage = cv2.drawContours(lettermask, letters, -1, (0, 255, 0), 1)
        cv2.imshow("Letter Image", cv2.cvtColor(letterimage, cv2.COLOR_RGB2BGR))

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
        ##NAVIGATION 

        ## Define the coordinates of the region of interest (ROI)
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 400, 1280, 410  # Adjust these coordinates as needed
        ## Default Resolution x = 320, y = 240

        # ## Crop the image to the ROI
        # roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        # cv2.waitKey(1)
        
        # hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
        # lower_white = np.array([0, 0, 200])
        # upper_white = np.array([255, 30, 255])
        # white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        # ## Define the lower and upper bounds for the color you want to detect (here, it's blue)
        # sensitivity = 15
        # cv2.waitKey(1)
        # ## Define a threshold value for detecting grayscale change
        # threshold_value = 100  # Adjust this threshold as needed

        # ## Find contours in the binary mask
        # pidcontours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cx = 0
        # cx = 0

        # pid_img = cv2.drawContours(frame, pidcontours, -1, (0, 255, 0), 1)

        # cxnet = 0
        # cynet = 0
        # moments = 0
        # cxavg = 640
        # ## Iterate through the contours and find the position of color change within the ROI
        # for contour in pidcontours:

        #     ## Calculate the centroid of the contour
        #     M = cv2.moments(contour)

        #     if M["m00"] != 0:
        #         cx = int(M["m10"] / M["m00"])

        #         ## Add the ROI offset to get the position within the original image
        #         cx += roi_x1
                
        #         cxnet += cx
        #         moments += 1

        #         #print(f"Position of color change within ROI: ({cx}, {cy})")
        # if moments != 0:
        #     cxavg = cxnet / moments

        # rate = rospy.Rate(2)
        # move = Twist()
        # move.linear.x = .1
        
        # cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)

        # if cxavg >= 0 and cxavg < 256:
        #     move.angular.z = 2
        #     cv2.putText(frame, str(cxavg)+" LEFT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # elif cxavg >= 256 and cxavg < 512:
        #     move.angular.z = 0.75
        #     cv2.putText(frame, str(cxavg)+" left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # elif cxavg >= 512 and cxavg < 728:
        #     move.angular.z = 0
        #     cv2.putText(frame, str(cxavg)+" Straight", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # elif cxavg >= 728 and cxavg < 1024:
        #     move.angular.z = -0.75
        #     cv2.putText(frame, str(cxavg)+" right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # else:
        #     move.angular.z = -2
        #     cv2.putText(frame, str(cxavg)+" RIGHT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

            
        # center_coordinates = (int(cxavg), int(h))  # Change the coordinates as needed
        # #print (center_coordinates)
        # radius = 30
        # color = (0, 0, 255)  # Red color in BGR format
        # thickness = -1 # Thickness of the circle's border (use -1 for a filled circle)
        # # Process the frame here (you can add your tracking code or other operations)
        # frame_with_circle = cv2.circle(frame, center_coordinates, radius, color, thickness)



        # cv2.imshow("PID", cv2.cvtColor(frame_with_circle, cv2.COLOR_RGB2BGR))
        # self.move_pub.publish(move)
        
        
    
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