#!/usr/bin/env python

import rospy
import numpy as np
from predict_update_og.msg import prior_liklihood_posterior, measurement_msg
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
from cv_bridge import CvBridge


import matplotlib.pyplot as plt

initialized = False
initial_data = None
pub = None

# global GRID_SIZE
GRID_SIZE = None
resolution = 1

bridge = CvBridge()



###################################################################################
def update_measurement(OG, likelihood):
    state = OG
    belief = np.zeros_like(OG)
    for i_ in range(GRID_SIZE[0] // resolution):
        for j in range(GRID_SIZE[1] // resolution):
            prior_fire = state[i_, j, 1]
            prior_Q = state[i_, j, 0]

            posterior_Q = (prior_Q * likelihood[i_, j, 0]) / (
                    prior_Q * likelihood[i_, j, 0] + (1 - prior_Q) * (1 - likelihood[i_, j, 0]))

            posterior_fire = (prior_fire * likelihood[i_, j, 1]) / (
                    prior_fire * likelihood[i_, j, 1] + (1 - prior_fire) * (1 - likelihood[i_, j, 1]))

            belief[i_, j, 0] = posterior_Q
            belief[i_, j, 1] = posterior_fire

    belief = np.clip(belief, 0, 1)
    print(np.argwhere(np.isnan(belief)))

    return belief


processed_data = []

data1 = None
data2 = None


###################################################################################
def callback1(data):
    global data1
    bridge = CvBridge()
    
    data1 = bridge.imgmsg_to_cv2(data.image, desired_encoding='passthrough')
    msg_header = data.header
    msg_annotation = data.json_annotation

    data1 = np.asarray(data1)   

    rospy.loginfo("Received measurements: \n%s", data1.shape)

    data1 = np.expand_dims(data1, axis= -1)   
    process_and_publish_result()


def callback2(data):

    global initialized, initial_data, GRID_SIZE, processed_data, data2


    if not initialized:
        rospy.loginfo("Initializing...")
        # Process and store the initial data
        height = data.height
        width = data.width
        depth = data.depth
        data2 = np.ones((height, width, depth))  * 0.5 # Initialize with zeros
        initialized = True
        rospy.loginfo("Initialization complete.")
    else:

        # Receive data from the prediction function (prior)
        height = data.height
        width = data.width
        depth = data.depth
        data2 = np.array(data.state).reshape(height, width, depth)
        rospy.loginfo("Received prior: \n%s", data2.shape)

        likelihood = data2

        GRID_SIZE = (height, width, depth)
        process_and_publish_result()

def process_and_publish_result():
    global GRID_SIZE, processed_data, data2
    if data1 is not None and data2 is not None:
       posterior = update_measurement(data2, data2)
       processed_data.append(posterior)
       #####################################################################     
       rospy.loginfo('Posterior \n%s', posterior[:, :, -1].shape)
       #####################################################################
       # Publish the processed data back to the prediction function (to predict next state)
       array_msg = prior_liklihood_posterior()
       array_msg.height = data2.shape[0]
       array_msg.width = data2.shape[1]
       array_msg.depth = data2.shape[2]
       array_msg.state = posterior.flatten().tolist()
       pub.publish(array_msg)
       rospy.loginfo("Published Belief (Posterior): \n%s", posterior.shape)
    
    



def second_node():
    global pub
    rospy.init_node('update_node', anonymous=True)    
    ######################################################################
    rospy.Subscriber('prior_array_topic', prior_liklihood_posterior, callback2)
    rospy.Subscriber('georef_image', measurement_msg, callback1)
    #########################################################################
    global pub
    pub = rospy.Publisher('belief_array_topic', prior_liklihood_posterior, queue_size=10)

    rate = rospy.Rate(1)  # 1 Hz
    
    # Wait for initialization before starting the cyclic loop
    while not initialized and not rospy.is_shutdown():
        rate.sleep()

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    processed_data = []
    try:
        second_node()
    except rospy.ROSInterruptException:
        pass
