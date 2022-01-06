#!/usr/bin/env python
import time
from collections import namedtuple

# from typing import List, Set, Any
from queue import PriorityQueue

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
    cv.circle(img, tuple(p), 2, color, cv.FILLED, cv.LINE_AA)

def distance(a,b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def is_same_edge(e1,e2):
    return (e1.p1 == e2.p1 and e1.p2 == e2.p2) or (e1.p1 == e2.p2 and e1.p2 == e2.p1)

class CleaningBlocks:

    def __init__(self, occ_map):
        self.triangles = None
        self.occ_map = occ_map
        self.map_size = occ_map.shape
        self.rect = (0, 0, self.map_size[1], self.map_size[0])
        self.graph = Graph()

        # find corners
        ret, thresh = cv.threshold(occ_map, 90, 255, 0)
        thresh = np.uint8(thresh)
        self.map_rgb = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        # map_img_th = thresh.copy()
        # im2, contours, hierarchy = cv.findContours(map_img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(map_img_th, contours, -1, (255, 255, 255), 3)
        corners = cv.goodFeaturesToTrack(thresh, 25, 0.01, 10)
        self.corners = np.int0(corners)

        # find triangles
        # Create an instance of Subdiv2D
        self.sub_div = cv.Subdiv2D(self.rect)
        # Insert points into sub_div
        self.sub_div.insert(corners)
        # Filter triangles outside the polygon
        self.extract_triangles()

        # Build graph
        self.graph(len(self.triangles))

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
                    center = (np.round((t[0] + t[2] + t[4]) / 3), np.round((t[1] + t[3] + t[5]) / 3))
                    mat = np.array([[t[0], t[1], 1], [t[2], t[3], 1], [t[4], t[5], 1]])
                    area = np.linalg.det(mat) / 2

                    tri_edges = [[pt1,pt2,distance(pt1,pt2)], [pt1,pt3,distance(pt1,pt3)], [pt2,pt3,distance(pt2,pt3)]]
                    center_edges = [[pt1,center,distance(pt1,center)], [pt1,center,distance(pt1,center)], [pt2,center,distance(pt2,center)]]
                    filtered_triangles.append(Triangle(t, center, area))
                    self.graph.add_edges(tri_edges)
                    self.graph.add_edges(center_edges)

        self.triangles = filtered_triangles

    def get_triangles(self):
        return self.triangles

    # Draw delaunay triangles
    def draw_triangles(self, delaunay_color):
        img = self.map_rgb
        # Draw points
        for p in self.corners:
            draw_point(img, p[0], (0, 0, 255))

        for triangle in self.triangles:
            t = triangle.coordinates
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
            # print(triangle)
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

    def locate_initial_pose(self, first_pose):
        min_dist = 1000
        ind = 0
        x = first_pose[0]
        y = first_pose[1]
        for (i, triangle) in enumerate(self.triangles):
            c = triangle.center
            dist = distance(c ,first_pose)
            if dist < min_dist:
                min_dist = dist
                ind = i
        # closest_tri = self.triangles[ind]
        self.graph.add_starting_point(ind)

    def is_neighbor(self, v_i, u_i):
        v_edges = self.triangles[v_i].edges
        u_edges = self.triangles[u_i].edges
        for e1 in v_edges:
            for e2 in u_edges:
                if is_same_edge(e1, e2):
                    return True
        return False

    def add_vertices(self):
        min_dist = 1000
        ind = 0

        num_tri = len(self.triangles)
        for (i, triangle) in enumerate(self.triangles):
            c1 = triangle.center

            for j in range(i+1, num_tri ):
                if neighbors_triangles(i, j):
                    c2 = self.triangles[j].center
                    dist = distance(c1, c2)
            if dist < min_dist:
                min_dist = dist
                ind = i
        # closest_tri = self.triangles[ind]
        self.graph.add_starting_point(ind)

    def sort(self, first_pose):

        self.locate_initial_pose(first_pose)

        compute_dist_mat()

        t = closest_tri.coordinates
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        img = self.map_rgb
        cv.line(img, pt2, pt3, (255, 0, 0), 1, cv.LINE_AA, 0)
        cv.line(img, pt3, pt1, (255, 0, 0), 1, cv.LINE_AA, 0)
        cv.line(img, pt1, pt2, (255, 0, 0), 1, cv.LINE_AA, 0)
        draw_point(img, np.uint32(closest_tri.center), (255, 255, 0))
        draw_point(img, np.uint32(first_pose), (0, 255, 255))
        cv.imshow('delaunay', self.map_rgb)
        cv.waitKey(0)


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

    def get_first_pose(self):
        while self.initial_pose is None:
            time.sleep(1)

        pos = np.array([self.initial_pose.position.x, self.initial_pose.position.y])
        return self.position_to_map(pos)

class Graph:
    def __init__(self, num_of_vertices):
        self.v = num_of_vertices
        self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]
        self.visited = []

    def add_edges(self, edges):
        for e in edges:
            self.add_edge(e[0],e[1],e[2])

    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight

    def dijkstra(graph, start_vertex):
        D = {v: float('inf') for v in range(graph.v)}
        D[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))

        while not pq.empty():
            (dist, current_vertex) = pq.get()
            graph.visited.append(current_vertex)

            for neighbor in range(graph.v):
                if graph.edges[current_vertex][neighbor] != -1:
                    dist = graph.edges[current_vertex][neighbor]
                    if neighbor not in graph.visited:
                        old_cost = D[neighbor]
                        new_cost = D[current_vertex] + dist
                        if new_cost < old_cost:
                            pq.put((new_cost, neighbor))
                            D[neighbor] = new_cost
        return D
#
# class Graph:
#     def __init__(self, gdict=None):
#         if gdict is None:
#             gdict = {}
#         self.gdict = gdict
#
#         self.start_ind = 0
#
#     def add_starting_point(self, start):
#         self.start_ind = start
#
#     def edges(self):
#         return self.findedges
#
#     # Add the new edge
#     def add_edge(self, edge):
#         edge = set(edge)
#         (vrtx1, vrtx2) = tuple(edge)
#         if vrtx1 in self.gdict:
#             self.gdict[vrtx1].append(vrtx2)
#         else:
#             self.gdict[vrtx1] = [vrtx2]
#
#     def add_edges(self, edges):
#         for e in edges:
#             self.add_edge(e)
#
#     # List the edge names
#     def find_edges(self):
#         edge_name = []  # type: List[Set[Any]]
#         for vrtx in self.gdict:
#             for nxtvrtx in self.gdict[vrtx]:
#                 if {nxtvrtx, vrtx} not in edge_name:
#                     edge_name.append({vrtx, nxtvrtx})
#
#         return edge_name
#
#     def get_vertices(self):
#         return list(self.gdict.keys())
#
#     # Add the vertex as a key
#     def add_vertex(self, vrtx):
#         if vrtx not in self.gdict:
#             self.gdict[vrtx] = []
#
#     def add_vertices(self, vertices):
#         for (i, v) in enumerate(vertices):
#             self.add_vertex(i)



def vacuum_cleaning():
    print('start vacuum_cleaning')
    raise NotImplementedError


def inspection():
    print('start inspection')
    raise NotImplementedError

def

# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('get_map_example')
    ms = MapService()
    Edge =  namedtuple('Edge', ['p1','p2'])
    Triangle = namedtuple('Triangle', ['coordinates', 'center', 'area', 'edges'])

    cb = CleaningBlocks(ms.map_arr)

    first_pose = ms.get_first_pose()
    cb.sort(first_pose)

    triangle_list = cb.get_triangles()
    # Draw delaunay triangles
    cb.draw_triangles((0, 255, 0))

    triangles = []  # Tom's format
    for triangle in triangle_list:
        t = triangle.coordinates
        triangles.append((np.array((t[0], t[1], 0)), np.array((t[2], t[3], 0)), np.array((t[4], t[5], 0))))

    exec_mode = sys.argv[1]
    print('exec_mode:' + exec_mode)

    # For anyone who wants to change parameters of move_base in python, here is an example:
    # rc_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
    # rc_DWA_client.update_configuration({"max_vel_x": "np.inf"})

    if exec_mode == 'cleaning':
        vacuum_cleaning()
    elif exec_mode == 'inspection':
        inspection()
    else:
        print("Code not found")
        raise NotImplementedError
