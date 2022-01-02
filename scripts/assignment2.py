#!/usr/bin/env python

import rospy
import sys
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.srv import GetMap

import matplotlib.pyplot as plt

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
import dynamic_reconfigure.client


class CostmapUpdater:
    def __init__(self):
        self.cost_map = None
        self.shape = None
        rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.init_costmap_callback)
        rospy.Subscriber('/move_base/global_costmap/costmap_updates', OccupancyGridUpdate, self.costmap_callback_update)

    def init_costmap_callback(self, msg):
        print('only once')  # For the student to understand
        self.shape = msg.info.height, msg.info.width
        self.cost_map = np.array(msg.data).reshape(self.shape)

    def costmap_callback_update(self, msg):
        print('periodically')  # For the student to understand
        shape = msg.height, msg.width
        data = np.array(msg.data).reshape(shape)
        self.cost_map[msg.y:msg.y + shape[0], msg.x: msg.x + shape[1]] = data
        self.show_map()  # For the student to see that it works

    def show_map(self):
        if not self.cost_map is None:
            plt.imshow(self.cost_map)
            plt.show()


class MapService(object):

    def __init__(self):
        """
        Class constructor
        """
        rospy.wait_for_service('static_map')
        static_map = rospy.ServiceProxy('static_map', GetMap)
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


#if __name__ == '__main__':
#    rospy.init_node('get_map_example')
#    ms = MapService()
#    ms.show_map(ms.position_to_map(np.array([0.0, 0.0])))
#    ms.show_map(ms.position_to_map(np.array([0.5, -3.3])))
#    print(ms.resolution)

#For anyone who wants to change parameters of move_base in python, here is an example:
rc_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
rc_DWA_client.update_configuration({"max_vel_x": "np.inf"})

def initial_pose(self,msg):
self.initialpose = msg.pose.pose
print("initial pose is")
print("X=" + str(self.initalpose.position.x))
print("y=" + str(self.initalpose.position.y))

def vacuum_cleaning():
    print('start vacuum_cleaning')
    raise NotImplementedError

def inspection():
    print('start inspection')
    raise NotImplementedError


def inspection_advanced():
    print('start inspection_advanced')
    raise NotImplementedError



# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':

    exec_mode = sys.argv[1] 
    print('exec_mode:' + exec_mode)
    
    rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initial_pose)
#    rospy.init_node('get_map_example')
#    ms = MapService()
#    ms.show_map(ms.position_to_map(np.array([0.0, 0.0])))
#    ms.show_map(ms.position_to_map(np.array([0.5, -3.3])))
#    print(ms.resolution)
    if exec_mode == 'cleaning':
        vacuum_cleaning()
    elif exec_mode == 'inspection':
        inspection()
    else:
        print("Code not found")
        raise NotImplementedError
