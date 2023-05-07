""" A set of robotics control functions """

import numpy as np
from operator import itemgetter

# def reactive_obst_avoid(lidar):
#     """
#     Simple obstacle avoidance
#     lidar : placebot object with lidar data
#     """
#     # TODO for TP1

#     values = lidar.get_sensor_values()
#     angles = lidar.get_ray_angles()
#     #print(lidar.max_range)

    
#     #print(values)
  
#     #print(d)

#     #print(values)
#     #print(angles[181:len(angles)])
#     #for i in range(181,len(angles)):
#     #    d = 
#     #print(angles)


#     d_max = np.max(values)
#     d_min = np.min(values)
#     sum = 0
#     for i in range(178, 182):
#         sum = sum + values[i]
#     d_mean = sum/5
#     forward = (values[180]-d_min)/(d_max)
#     #print(forward)
#     idx_max = np.argmax(values)
#     #print(angles[idx_max])
#     #rotation = (angles[idx_max]/pi)

#     #rotation = random.randint(0, 360)
#     i_max = np.argmax(values)
#     if forward < 0.2:
#         if i_max < 180:
#             rotation = random.randint(-10,0)/10
#         else:
#             rotation = random.randint(0,10)/10  
#     else:
#         rotation = 0


    

#    # if forward < 0.1:
#         #rotation = random.randint(-10, 10)/10
#     #    rotation = random.randint(-1,0)
#     #else:
#     #    rotation = 0


#     command = {"forward": forward,
#                "rotation": rotation}

#     return command

def findWall(lidar):
    values = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    # frontal cone
    filter = abs(angles) < np.pi/6
    front = values[filter]

    # lateral cone on the right
    filter = abs(angles+np.pi/2) < np.pi/10
    right = values[filter]

    # finds a wall
    if np.min(front) > 40:
        command = {"forward": 0.2,
            "rotation": 0}
        return command, False
    
    # wall in front
    if np.min(right) > 20:
        command = {"forward": 0,
            "rotation": 0.1}
        return command, False
    else :
        # we're done here
        command = {"forward": 0,
            "rotation": 0}
        return command, True

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TP1


    values = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    # frontal cone
    filter = abs(angles) < np.pi/6
    front = values[filter]

    # lateral cone on the right
    filter = abs(angles+np.pi/2) < np.pi/8
    right = values[filter]

    turn = -0.3
    speed = 0.1

    margin = 20
    dx = np.min(right) - margin

    # wall in front
    if np.min(front) < 35:
        turn = 0.3
        speed = 0
        #print("wall in front")
    # if far from right wall, get closer
    # elif dx > 5:
    #     turn = -0.2
    #     speed = 0.1
    #     print("aproxaing to the right")
    # follow right wall
    elif np.mean(right) < 30:
        k = 0.01
        turn = -k*dx
        turn = np.clip(turn,-0.3,0.3)
        speed = 0.1
        #print("controle")
    #print(dx,turn)
    command = {"forward": speed,
                "rotation": turn}
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
    obst_lidar = min(cart.T, key=itemgetter(0))
    obst = [0,0,0]
    obst[0] = pose[0] + obst_lidar[0] * np.cos(obst_lidar[1]+pose[2])
    obst[1] = pose[1] + obst_lidar[0] * np.sin(obst_lidar[1]+pose[2])
    repul_vec = obst - pose
    grad_obs = (1/np.linalg.norm(repul_vec[:2]))*repul_vec[:2]

    # attraction au objectif
    K = 1
    obj_vec = goal - pose
    # normalization
    grad_pos = (1/np.linalg.norm(obj_vec[:2]))*obj_vec[:2]


    # somme vecteurs et calcul rotation
    #fear = 0.5 # in %
    K_obst = 0.35*(20/(obst_lidar[0])**1 - 0.1)
    K_obst = 0 if K_obst < 0 else K_obst
    K_obst = 1 if K_obst > 1 else K_obst
    print(K_obst)
    fear = K_obst
    tot_vec = (1-fear)*grad_pos - (fear)*grad_obs
    print(tot_vec)
    #print('importance obstacle',(fear)*K_obst/((1-fear)+(fear)*K_obst))
    grad_tot = (1/np.linalg.norm(tot_vec[:2]))*tot_vec[:2]

    delta_rot = np.arctan2(grad_tot[1],grad_tot[0]) - pose[2]
    Krot = 1
    rot = Krot * delta_rot

    speed = np.linalg.norm(tot_vec[:2])
    speed = 0.1 + 0.2*speed -rot
    speed = np.clip(speed,0,0.3)

    # reglage limites comandes
    if rot < -1:
        rot = -1
    if rot > 1:
        rot = 1
    # if abs(rot) < 0.01:
    #     rot = 0

    if np.linalg.norm(obj_vec) < 20:
        command = {"forward": 0,
               "rotation": 0}
        flag = True
    else:
        command = {"forward": speed,
                   "rotation": rot}
        flag = False

    return command, flag
