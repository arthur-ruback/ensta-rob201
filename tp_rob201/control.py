""" A set of robotics control functions """

import numpy as np
import random as rd
import time 
from operator import itemgetter


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TP1

    values = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    # # repaire x devant et y sur la gauche
    # x = values * np.cos(angles)
    # y = values * np.sin(angles)

    filter = abs(angles) < np.pi/6
    obstacles = values[filter]
    obstacles = obstacles[obstacles < 50]

    if len(obstacles) > 0: # not empty
        command = {"forward": 0,
               "rotation": 0.3}
    else:
        command = {"forward": 0.3,
                "rotation": -0.1}

    return command

def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
   # TODO for TP2

    values = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    cart = np.vstack((values,angles))

    # constants de reglage
    SECURITY_DIST = 100

    # # repulsion obstacle plus proche
    # obst = min(cart.T, key=itemgetter(0))
    # print(obst)
    # vec_repul = 

    # attraction au objectif
    K = 1
    dist_vec = goal - pose
    grad_pos = (K/np.linalg.norm(dist_vec[:2]))*dist_vec[:2]

    delta_rot = np.arctan2(grad_pos[1],grad_pos[0]) - pose[2]
    Krot = 1
    rot = Krot * delta_rot

    

    # reglage limites comandes
    if rot < -1:
        rot = -1
    if rot > 1:
        rot = 1
    if abs(rot) < 0.01:
        rot = 0

    if np.linalg.norm(dist_vec) < 20:
        print('we are done here')
        command = {"forward": 0,
               "rotation": 0}
    else:
        command = {"forward": 0.1*np.linalg.norm(dist_vec[:2]),
                   "rotation": rot}

    return command
