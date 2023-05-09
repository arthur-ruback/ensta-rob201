""" A simple robotics navigation code including SLAM, exploration, planning"""

import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
import heapq
import math
from collections import defaultdict
import copy

PROB_SOIL = 4
prob_forte = 0.4
prob_faible = -0.05
N_it_loc = 150
var_x = 3
var_y = 3
var_ang = 0.01

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(
            self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros(
            (int(self.x_max_map), int(self.y_max_map)))

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """
        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def _conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """
        x_world = self.x_min_world + x_map *  self.resolution
        y_world = self.y_min_world + y_map *  self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world

    def add_map_line(self, x_0, y_0, x_1, y_1, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        MODIF1 : dont change the final value of the cell
        MODIF2 : add value 0 to cell avant as a way of interpolating 
        """
        # convert to pixels
        x_start, y_start = self._conv_world_to_map(x_0, y_0)
        x_end, y_end = self._conv_world_to_map(x_1, y_1)

        if x_start < 0 or x_start >= self.x_max_map or y_start < 0 or y_start >= self.y_max_map:
            return

        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return

        flag_swap = False
        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
            flag_swap = True
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
            flag_swap = True
        dx = x_end - x_start # recalculate differentials
        dy = y_end - y_start # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        if flag_swap == False:
            for x in range(x_start, x_end):
                coord = [y, x] if is_steep else [x, y]
                points.append(coord)
                error -= abs(dy)
                if error < 0:
                    y += y_step
                    error += dx
            points = np.array(points).T
        else:
            for x in range(x_start, x_end):
                coord = [y, x] if is_steep else [x, y]
                points.append(coord)
                error -= abs(dy)
                if error < 0:
                    y += y_step
                    error += dx
            points = np.array(points).T

        # add value to the points
        if len(points) != 0: 
            self.occupancy_map[points[0], points[1]] += val

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """
        x_px, y_px = self._conv_world_to_map(points_x, points_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.x_max_map),
                                np.logical_and(y_px >= 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val

    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        values = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        # filter lidar max values
        mask = values < lidar.max_range
        values = values[mask]
        angles = angles[mask]

        # get updated pose
        #pose = self.get_corrected_pose(pose)

        # repaire du robot: x devant et y sur la gauche
        x = pose[0] + values * np.cos(angles+pose[2])
        y = pose[1] + values * np.sin(angles+pose[2])

        # get nearest grid point in the map and add its value to score
        x_px, y_px = self._conv_world_to_map(x,y)
        select = np.logical_and(np.logical_and(x_px > 0, x_px < self.x_max_map),
                                np.logical_and(y_px > 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        score = np.sum(np.exp(self.occupancy_map[x_px, y_px]))
        
        return score

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in global frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4

        # treats case of second argument null
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        # angle global = angle repaire + angle mesure
        corrected_pose = [0,0,odom[2]+odom_pose_ref[2]]
        d = np.linalg.norm(odom[:2])
        alpha = np.arctan2(odom[1],odom[0])
        
        # corrected position
        #               x = xr               + d * cos(alpha + theta_0)
        corrected_pose[0] = odom_pose_ref[0] + d * np.cos(alpha+odom_pose_ref[2])
        #               y = yr               + d * sin(alpha + theta_0)
        corrected_pose[1] = odom_pose_ref[1] + d * np.sin(alpha+odom_pose_ref[2])

        return corrected_pose

    def localise(self, lidar, odom):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        # score with current reference
        best_pose_ref  = self.odom_pose_ref
        best_score = self.score(lidar, self.get_corrected_pose(odom,best_pose_ref))
        # print('Score avant :', best_score)
        # print('Pose avant    :', best_pose_ref)

        # generates random perturbations to the pose
        
        # tries with random positions
        it = 0
        while it < N_it_loc:
            delta = np.array([np.random.normal(0, var_x, 1), 
                    np.random.normal(0, var_y, 1),
                    np.random.normal(0, var_ang, 1)])
            pose_it = self.get_corrected_pose(odom,best_pose_ref+delta.T[0])
            score = self.score(lidar, pose_it)
            it += 1

            # updates
            if score > best_score:
                best_score = score
                best_pose_ref = best_pose_ref + delta.T[0]
                it = 0

        # updates best position and returns
        #print(count,best_score,best_pose,self.odom_pose_ref)
        self.odom_pose_ref = best_pose_ref

        # print('Score best  :', best_score)
        # print('Pose nouvelle :', best_pose_ref)

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3


        values = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        # repaire du robot: x devant et y sur la gauche
        x = pose[0] + values * np.cos(angles+pose[2])
        y = pose[1] + values * np.sin(angles+pose[2])

        # corrige le repaire avec la pose du robot

        # dans le repaire absolut (ou au moins ce qu'il pense le robot)
        points = np.vstack((x,y))

        # only the values that are obstacles
        mask = values < lidar.max_range-10
        x = x[mask]
        y = y[mask]

        self.add_map_points(x,y,prob_forte)

        values = lidar.get_sensor_values()
        values -= 10
        angles = lidar.get_ray_angles()

        # repaire du robot: x devant et y sur la gauche
        x = pose[0] + values * np.cos(angles+pose[2])
        y = pose[1] + values * np.sin(angles+pose[2])

        # corrige le repaire avec la pose du robot

        # dans le repaire absolut (ou au moins ce qu'il pense le robot)
        points = np.vstack((x,y))

        for pair in points.T:
            self.add_map_line(pose[0],pose[1],pair[0],pair[1],prob_faible)

        # seuillage
        self.occupancy_map[self.occupancy_map > PROB_SOIL] = PROB_SOIL
        self.occupancy_map[self.occupancy_map < -PROB_SOIL] = -PROB_SOIL

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """

        start = self._conv_world_to_map(start[0],start[1])
        goal = self._conv_world_to_map(goal[0],goal[1])

        occupancy_map = copy.deepcopy(self.occupancy_map)
        kernel = np.ones((15,15), dtype=int)
        occupancy_map[occupancy_map < 0] = 0
        occupancy_map = cv2.filter2D(occupancy_map, -1, kernel)

        cv2.imshow("occupancy map", occupancy_map)

        # min heap to contain values to explore next
        openset = [(0,start)]
        heapq.heapify(openset)
        
        # dictionary to trace back route
        cameFrom = {}

        # cost to get to each cell
        gScore = defaultdict(lambda:math.inf)
        gScore[start] = 0

        # best guess of cost for each cell 
        fScore = defaultdict(lambda:math.inf)
        fScore[start] = self.heuristic(start,goal)

        while len(openset) > 0:
            current = heapq.heappop(openset)
            if current[1] == goal:
                return self.reconstructPath(cameFrom, goal)
            
            neighbours = self.get_neighbors(current[1], occupancy_map)
            for cell in neighbours:
                tentativeGScore = gScore[current[1]] + self.heuristic(current[1],cell)
                if tentativeGScore < gScore[cell]:
                    # better path, recording it
                    cameFrom[cell] = current[1]
                    gScore[cell] = tentativeGScore
                    fScore[cell] = tentativeGScore + self.heuristic(cell,goal)
                    if cell not in openset:
                        heapq.heappush(openset,(fScore[cell], cell))

        # goal was never reached
        print('failed getting to objective')
        return [start]

    def display(self, robot_pose,):
        """
        Screen display of map and robot pose, using matplotlib
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world, self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(robot_pose[0], robot_pose[1], delta_x, delta_y,
                  color='red', head_width=5, head_length=10, )

        # plt.show()
        plt.pause(0.001)

    def display2(self, robot_pose, trajectory = []):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        
        traj_world = []
        for x,y in trajectory:
            traj_world.append((self._conv_map_to_world(x,y)))
        trajectory = []
        for x,y in traj_world:
            trajectory.append((self._conv_world_to_map(x,-y)))
        for i in range(len(trajectory)-1):
            cv2.line(img2, trajectory[i], trajectory[i+1], (255, 128, 0), 1)

        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

    def save(self, filename, pose):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world,
                           self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + '.png')

        with open(filename + ".p", "wb") as fid:
            pickle.dump({'occupancy_map': self.occupancy_map,
                         'resolution': self.resolution,
                         'x_min_world': self.x_min_world,
                         'x_max_world': self.x_max_world,
                         'y_min_world': self.y_min_world,
                         'y_max_world': self.y_max_world,
                         'my_pose': pose}, fid)

    def load(self, filename, pose):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        with open(filename + ".p", "wb") as fid:
            [self.occupancy_map,
                self.resolution,
                self.x_min_world,
                self.x_max_world,
                self.y_min_world,
                self.y_max_world,
                pose] = pickle.load(fid)
            
        # TODO: correct data from pose

    def get_neighbors(self, current, occupancy_map):
        li = []
        # iterate through neighbors
        for ip in range(-1,2,1):
            for jp in range(-1,2,1):
                if not((ip == 0) and (jp == 0)):
                    i = ip + current[0]
                    j = jp + current[1]
                    # if empty
                    if occupancy_map[i,j] < 0.1:
                        li.append((i,j))
        return li
    
    def heuristic(self, a, b):
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
                        
    def reconstructPath(self, cameFrom, current):
        totalPath = [current]
        while current in cameFrom.keys():
            current = cameFrom[current]
            totalPath.insert(0,current)
        return totalPath


