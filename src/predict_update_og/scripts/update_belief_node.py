#!/usr/bin/env python

import rospy
import numpy as np
from predict_update_og.msg import prior_liklihood_posterior
import matplotlib.pyplot as plt

initialized = False
initial_data = None
pub = None

global GRID_SIZE

resolution = 1


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


###################################################################################

def callback(data):
    global initialized, initial_data, pub, GRID_SIZE, processed_data

    if not initialized:
        rospy.loginfo("Initializing...")

        height = data.height
        width = data.width
        depth = data.depth
        initial_data = np.zeros((height, width, depth))  # Initialize with zeros
        initialized = True
        rospy.loginfo("Initialization complete.")
    else:


        height = data.height
        width = data.width
        depth = data.depth
        prior_state = np.array(data.state).reshape(height, width, depth)
        rospy.loginfo("Received Array: \n%s", prior_state.shape)

        likelihood = prior_state

        GRID_SIZE = (height, width, depth)

        posterior = update_measurement(prior_state, likelihood)

        processed_data.append(posterior)

        #####################################################################
        plt.imshow(posterior[:, :, -1])
        plt.show()
        plt.close()
        #####################################################################
        # Publish the processed data back to the prediction function (to predict next state)
        array_msg = prior_liklihood_posterior()
        array_msg.height = height
        array_msg.width = width
        array_msg.depth = depth
        array_msg.state = posterior.flatten().tolist()
        pub.publish(array_msg)
        rospy.loginfo("Published Array: \n%s", posterior.shape)


def second_node():
    global pub
    rospy.init_node('update_node', anonymous=True)
    pub = rospy.Publisher('belief_array_topic', prior_liklihood_posterior, queue_size=10)
    rospy.Subscriber('prior_array_topic', prior_liklihood_posterior, callback)
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
