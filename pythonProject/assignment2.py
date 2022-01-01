#!/usr/bin/env python

import rospy
import sys
import numpy as np
import cv2 as cv
from nav_msgs.srv import GetMap

import matplotlib.pyplot as plt

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
import dynamic_reconfigure.client


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv.circle(img, tuple(p[0]), 2, color, cv.FILLED, cv.LINE_AA)


def line_in_room(map, mid_p_i):
    if map[mid_p_i] != -1:  # mid pixel
        return True

    size = map.shape

    if mid_p_i[0] > 0:
        if map[(mid_p_i[0] - 1, mid_p_i[1])] != -1:  # left pixel
            return True
        if mid_p_i[1] > 0:  # left up pixel
            if map[(mid_p_i[0] - 1, mid_p_i[1] - 1)] != -1:
                return True
        if mid_p_i[1] < size[0] - 1:  # left down pixel
            if map[(mid_p_i[0] - 1, mid_p_i[1] + 1)] != -1:
                return True

    if mid_p_i[0] < size[1] - 1:
        if map[(mid_p_i[0] + 1, mid_p_i[1])] != -1:  # right pixel
            return True
        if mid_p_i[1] > 0:  # left up pixel
            if map[(mid_p_i[0] + 1, mid_p_i[1] - 1)] != -1:
                return True
        if mid_p_i[1] < size[0] - 1:  # left down pixel
            if map[(mid_p_i[0] + 1, mid_p_i[1] + 1)] != -1:
                return True

    if mid_p_i[1] > 0 and \
            map[(mid_p_i[0], mid_p_i[1] - 1)] != -1:  # up pixel
        return True

    if mid_p_i[1] < size[0] - 1 and \
            map[(mid_p_i[0], mid_p_i[1] + 1)] != -1:  # down pixel
        return True

    return False


# Draw delaunay triangles
def draw_delaunay(img_th, map, subdiv, delaunay_color, draw):
    triangle_list = subdiv.getTriangleList()

    size = img_th.shape
    r = (0, 0, size[1], size[0])
    filtered_triangles = []
    for t in triangle_list:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        mid1 = (np.uint32((t[1] + t[3]) / 2), np.uint32((t[0] + t[2]) / 2))
        mid2 = (np.uint32((t[3] + t[5]) / 2), np.uint32((t[2] + t[4]) / 2))
        mid3 = (np.uint32((t[1] + t[5]) / 2), np.uint32((t[0] + t[4]) / 2))
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            if line_in_room(map, mid1) and line_in_room(map, mid2) and line_in_room(map, mid3):
                if draw is True:
                    cv.line(img_th, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
                    cv.line(img_th, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)
                    cv.line(img_th, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)

            filtered_triangles.append(t)

    return filtered_triangles


class CostMapUpdater:

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


# For anyone who wants to change parameters of move_base in python, here is an example:
# rc_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
# rc_DWA_client.update_configuration({"max_vel_x": "np.inf"})


def initial_pose(self, msg):
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


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    #   rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initial_pose)
    rospy.init_node('get_map_example')
    ms = MapService()
    mapImg = ms.map_arr

    ret, thresh = cv.threshold(mapImg, 90, 255, 0)
    thresh = np.uint8(thresh)
    map_img_th = thresh.copy()
    map_rgb = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)

    im2, contours, hierarchy = cv.findContours(map_img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(map_img_th, contours, -1, (255, 255, 255), 3)

    corners = cv.goodFeaturesToTrack(mapImg, 25, 0.01, 10)
    corners = np.int0(corners)


    # Rectangle to be used with Subdiv2D
    size = mapImg.shape
    rect = (0, 0, size[1], size[0])
    # Create an instance of Subdiv2D
    subdiv = cv.Subdiv2D(rect)

    # Insert points into subdiv
    subdiv.insert(corners)

    # Draw points
    for p in corners:
        draw_point(map_rgb, p, (0, 0, 255))

    # Draw delaunay triangles
    triangle_list = draw_delaunay(map_rgb, mapImg, subdiv, (0, 255, 0), draw=True);
    triangles = []
    for t in triangle_list:
        triangles.append((np.array((t[0], t[1], 0)),np.array((t[2], t[3], 0)),np.array((t[4], t[5], 0))))

    # Show results
    cv.imshow('delaunay', map_rgb)
    cv.waitKey(0)

    exec_mode = sys.argv[1]
    print('exec_mode:' + exec_mode)

    if exec_mode == 'cleaning':
        vacuum_cleaning()
    elif exec_mode == 'inspection':
        inspection()
    else:
        print("Code not found")
        raise NotImplementedError
