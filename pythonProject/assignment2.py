#!/usr/bin/env python

import rospy
import sys
import numpy as np
import cv2 as cv

from geometry_msgs.msg import PoseWithCovarianceStamped
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


class CleaningBlocks:

    def __init__(self, occ_map):
        self.triangles = None
        self.occ_map = occ_map
        self.map_size = occ_map.shape
        self.rect = (0, 0, self.map_size[1], self.map_size[0])

        ret, thresh = cv.threshold(occ_map, 90, 255, 0)
        thresh = np.uint8(thresh)
        self.map_rgb = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        # map_img_th = thresh.copy()
        # im2, contours, hierarchy = cv.findContours(map_img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(map_img_th, contours, -1, (255, 255, 255), 3)
        corners = cv.goodFeaturesToTrack(thresh, 25, 0.01, 10)
        self.corners = np.int0(corners)

        # Create an instance of Subdiv2D
        self.sub_div = cv.Subdiv2D(self.rect)
        # Insert points into sub_div
        self.sub_div.insert(corners)

        self.extract_triangles()

    def extract_triangles(self):
        initial_triangles = self.sub_div.getTriangleList()

        r = self.rect
        filtered_triangles = []
        for t in initial_triangles:

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            mid1 = (np.uint32((t[1] + t[3]) / 2), np.uint32((t[0] + t[2]) / 2))
            mid2 = (np.uint32((t[3] + t[5]) / 2), np.uint32((t[2] + t[4]) / 2))
            mid3 = (np.uint32((t[1] + t[5]) / 2), np.uint32((t[0] + t[4]) / 2))
            if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
                if self.line_in_room(mid1) and self.line_in_room(mid2) and self.line_in_room(mid3):
                    filtered_triangles.append(t)

        self.triangles = filtered_triangles

    def get_triangles(self):
        return self.triangles

    # Draw delaunay triangles
    def draw_triangles(self, delaunay_color):
        img = self.map_rgb
        # Draw points
        for p in self.corners:
            draw_point(img, p, (0, 0, 255))

        for t in self.triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
            # cv.imshow('delaunay', img)
            # cv.waitKey(0)

        # Show results
        cv.imshow('delaunay', img)
        cv.waitKey(0)
        # cv.imwrite('delaunay.jpg', img)

    def line_in_room(self, mid_p_i):

        map = self.occ_map

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
        self.initial_pose = None
        rospy.wait_for_service('static_map')
        static_map = rospy.ServiceProxy('static_map', GetMap)
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.init_pose)

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

    def init_pose(self, msg):
        self.initial_pose = msg.pose.pose
        print("initial pose is")
        print("X=" + str(self.initial_pose.position.x))
        print("y=" + str(self.initial_pose.position.y))


# For anyone who wants to change parameters of move_base in python, here is an example:
# rc_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
# rc_DWA_client.update_configuration({"max_vel_x": "np.inf"})


def vacuum_cleaning():
    print('start vacuum_cleaning')
    raise NotImplementedError


def inspection():
    print('start inspection')
    raise NotImplementedError


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('get_map_example')
    ms = MapService()
    occ_map = ms.map_arr

    cb = CleaningBlocks(occ_map)
    triangle_list = cb.get_triangles()
    # Draw delaunay triangles
    cb.draw_triangles((0, 255, 0))

    triangles = []
    for t in triangle_list:
        triangles.append((np.array((t[0], t[1], 0)), np.array((t[2], t[3], 0)), np.array((t[4], t[5], 0))))

    exec_mode = sys.argv[1]
    print('exec_mode:' + exec_mode)

    if exec_mode == 'cleaning':
        vacuum_cleaning()
    elif exec_mode == 'inspection':
        inspection()
    else:
        print("Code not found")
        raise NotImplementedError
