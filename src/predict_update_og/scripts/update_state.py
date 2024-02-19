#!/usr/bin/env python

import rospy
import numpy as np
from predict_update_og.msg import prior_liklihood_posterior

def publisher():
    rospy.init_node('array_publisher', anonymous=True)
    pub = rospy.Publisher('array_topic', prior_liklihood_posterior, queue_size=15)
    rate = rospy.Rate(10)  # 1 Hz

    while not rospy.is_shutdown():
        # Generate a random multidimensional numpy array
        array_data = np.ones((10, 10, 3))  
        height, width, depth = array_data.shape

        
        # Create the message
        array_msg = prior_liklihood_posterior()
        array_msg.height = height
        array_msg.width = width
        array_msg.depth = depth
        array_msg.state = array_data.flatten().tolist()

        # Publish the message
        pub.publish(array_msg)
        rospy.loginfo("Published Array: \n%s", array_data)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass

