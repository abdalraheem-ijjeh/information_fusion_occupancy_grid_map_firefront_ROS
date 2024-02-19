#!/usr/bin/env python

import rospy
from predict_update_og.msg import prior_liklihood_posterior
import numpy as np


def callback(data):
    height = data.height
    width = data.width
    depth = data.depth
    array_data = np.array(data.state).reshape(height, width, depth)
    rospy.loginfo("Received Array: \n%s", array_data)


def subscriber():
    rospy.init_node('array_subscriber', anonymous=True)
    rospy.Subscriber('array_topic', prior_liklihood_posterior, callback)
    rospy.spin()


if __name__ == '__main__':
    subscriber()
