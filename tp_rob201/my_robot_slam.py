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
    
from control import reactive_obst_avoid, potential_field_control


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

        # Init SLAM object
        self._size_area = (800, 800)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        self.lastScore = 0

    def control(self):
        """
        Main control function executed at each time step
        """

        if self.counter == 0:
            self.tiny_slam.update_map(self.lidar(),self.odometer_values())
            score = self.tiny_slam.localise(self.lidar(),self.odometer_values())
        else:
            score = self.tiny_slam.localise(self.lidar(),self.odometer_values())
            print(score)
            if score > 20:
                self.tiny_slam.update_map(self.lidar(),self.tiny_slam.get_corrected_pose(self.odometer_values()))

        # affichage
        if self.counter % 10 == 0:
            self.tiny_slam.display2(self.tiny_slam.get_corrected_pose(self.odometer_values()))
            # print('     True position: ', [int(self.true_position()[0]), int(self.true_position()[1]), int(self.true_angle())])
            # print('Estimated position: ', self.odometer_values().astype(int)+[439, 200, 0])

        self.counter += 1

        #sleep(0.2)

        command = reactive_obst_avoid(self.lidar())

        return command


# Compute new command speed to perform obstacle avoidance
#objectif = np.array([-50,-200,1])
#command = potential_field_control(self.lidar(),self.odometer_values(),objectif)