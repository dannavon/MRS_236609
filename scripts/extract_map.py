#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.srv import GetMap


class MapService(object):

    def __init__(self):
        """
        Class constructor
        """
        print('wait for static_map')
        ag = 0
        rospy.wait_for_service('tb3_%d/static_map'%ag)
        print('get static_map')
        static_map = rospy.ServiceProxy('tb3_0/static_map', GetMap)
        self.map_data = static_map().map
        self.map_org = np.array([self.map_data.info.origin.position.x, self.map_data.info.origin.position.y])
        shape = self.map_data.info.height, self.map_data.info.width
        self.map_arr = np.array(self.map_data.data, dtype='float32').reshape(shape)
        self.resolution = self.map_data.info.resolution

    def show_map(self, point=None):
        plt.imshow(self.map_arr)
        if point is not None:
            plt.scatter([point[0]], [point[1]])
        plt.show()

    def position_to_map(self, pos):
        return (pos - self.map_org) // self.resolution

    def map_to_position(self, indices):
        return indices * self.resolution + self.map_org


if __name__ == '__main__':
    rospy.init_node('get_map_example')
    ms = MapService()
    ms.show_map(ms.position_to_map(np.array([0.0, 0.0])))
    ms.show_map(ms.position_to_map(np.array([0.5, -3.3])))
    print(ms.resolution)
