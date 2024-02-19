#!/usr/bin/env python

import rospy
import numpy as np
from predict_update_og.msg import prior_liklihood_posterior, measurement_msg
from scipy.spatial import distance
import random

initialized = False
initial_data = None

##########################################################
# Displacements from a cell to its eight nearest neighbours
neighbourhood = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# Forest size (number of cells in x and y directions).
nx, ny = 500, 500
# Initialize the forest grid.

predicted_state = np.ones((ny, nx, 2), dtype=np.float16) * 0.3

#######################################################
radius = 5
for i, j in zip(range(ny), range(nx)):
    if distance.euclidean((i, j), (ny // 2, nx // 2)) <= radius:
        print('YES')
        predicted_state[i, j, 1] = 0.9

#######################################################
global pub


##########################################
def transition_model(X_prev, noise_level):
    beta = random.randint(5, 10) / 10
    omega = random.randint(5, 10) / 10
    X_new = np.zeros_like(X_prev)
    for ix in range(nx):
        for iy in range(ny):
            R = 0
            temp_value_F_K_T = 0
            for y, x in neighbourhood:
                if (1 < iy < ny - 1) and (1 < ix < nx - 1):
                    F_j_t_1 = round(X_prev[iy + y, ix + x, 1])
                    if F_j_t_1 == 1 and round(X_prev[iy, ix, 0]) == 0:  # if Fj j,t−1 = 1 and j ∈ R(k) and Qk,t−1 = 0
                        temp_value_F_K_T += X_prev[iy + y, ix + x, 1]
                        R += 1
                    # if F_j_t_1 == 0 or round(X_prev[iy, ix, 0]) == 1:
                    #     X_new_temp[iy, ix, 1] = 0
            if R != 0:
                X_new[iy, ix, 1] = (1 - X_prev[iy, ix, 0]) * (
                        X_prev[iy, ix, 1] + ((1 / abs(R)) * omega * temp_value_F_K_T))
            else:
                X_new[iy, ix, 1] = (1 - X_prev[iy, ix, 0]) * (X_prev[iy, ix, 1])
            X_new[iy, ix, 0] = X_prev[iy, ix, 0] + (1 - X_prev[iy, ix, 0]) * (beta * X_prev[iy, ix, 1])

    X_new = np.clip(X_new, 0, 1)
    X_new = X_new + np.random.normal(scale=noise_level, size=X_new.shape)
    X_new = np.clip(X_new, 0, 1)
    return X_new


def callback(data):
    # global pub
    # Retrieve data from the message
    height = data.height
    width = data.width
    depth = data.depth
    array_data = np.array(data.state).reshape(height, width, depth)

    rospy.loginfo("Received Array: \n%s", array_data.shape)

    processed_data = transition_model(array_data, noise_level=0.0)

    # Publish the processed data back to the second node
    array_msg = prior_liklihood_posterior()
    array_msg.height = height
    array_msg.width = width
    array_msg.depth = depth
    array_msg.state = processed_data.flatten().tolist()
    pub.publish(array_msg)
    rospy.loginfo("Published Array: \n%s", processed_data.shape)


def first_node():
    global pub, initialized
    rospy.init_node('first_node', anonymous=True)
    pub = rospy.Publisher('prior_array_topic', prior_liklihood_posterior, queue_size=10)
    rospy.Subscriber('belief_array_topic', prior_liklihood_posterior, callback)

    rate = rospy.Rate(1)  # 1 Hz
    count = 0
    while count < 5 and not rospy.is_shutdown():
        count += 1
        data = predicted_state

        array_msg = prior_liklihood_posterior()
        array_msg.height, array_msg.width, array_msg.depth = data.shape
        array_msg.state = data.flatten().tolist()

        # Publish the message
        pub.publish(array_msg)
        rospy.loginfo("Published Array: \n%s", data.shape)
        rate.sleep()

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    try:
        first_node()
    except rospy.ROSInterruptException:
        pass
