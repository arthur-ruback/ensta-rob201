"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np
from time import sleep

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam
    
from control import reactive_obst_avoid, potential_field_control, findWall


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # objectif
        self.objectifList = [[-250,-400,1], [-400,-500,1], [-500,-370,1], [-210,-270,1], [-300,-240,1], [-500,-300,1], [-500,-120,1]]
        self.i = 0

        self.objectif = np.array([0,0,0])

        # Init SLAM object
        self._size_area = (1200, 1200)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        self.lastScore = 0

        self.flagWallFound = False

    def control(self):
        #print(self.odometer_values())
        """
        Main control function executed at each time step
        """
        # Compute new command speed to perform obstacle avoidance

        # self.objectif bon
        # self.objectif dur
        #self.objectif = np.array([-200,50,1])
        #print(self.counter)

        if self.counter < 2:
            self.tiny_slam.update_map(self.lidar(),self.odometer_values())
            score = self.tiny_slam.localise(self.lidar(),self.odometer_values())
            self.last_score = score
            
        else:
            score = self.tiny_slam.localise(self.lidar(),self.odometer_values())
            #print(score, 0.95*self.last_score)
            if score > 50:
                self.tiny_slam.update_map(self.lidar(),self.tiny_slam.get_corrected_pose(self.odometer_values()))
                self.last_score = score

        # affichage
        if self.counter % 1 == 0:
            self.tiny_slam.display2(self.tiny_slam.get_corrected_pose(self.odometer_values()))
            # print('     True position: ', [int(self.true_position()[0]), int(self.true_position()[1]), int(self.true_angle())])
            # print('Estimated position: ', self.odometer_values().astype(int)+[439, 200, 0])

        self.counter += 1

        #sleep(0.2)
        if (self.flagWallFound == False):
            command, self.flagWallFound = findWall(self.lidar())
            print("finding wall")
        else:
            command = reactive_obst_avoid(self.lidar())
        # flag_next_point = False
        # command, flag_next_point = potential_field_control(self.lidar(),self.odometer_values(),self.objectif)
        # if flag_next_point:
        #     if(self.i < len(self.objectifList)):
        #         self.objectif = self.objectifList[self.i]
        #         self.i += 1
        #     else:
        #         print("Done!")

        

        return command



'''

'''