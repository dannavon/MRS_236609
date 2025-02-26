#!/usr/bin/env python2.7
import rospy
import actionlib
import sys
import time
import math
import numpy as np
import dynamic_reconfigure.client
import cv2 as cv
import matplotlib.pyplot as plt
import tf
import logging

from scipy.spatial.transform import Rotation as R
# from scipy.misc import toimage
from scipy import ndimage
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
import dynamic_reconfigure.client
import std_msgs
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid, Odometry
from map_msgs.msg import OccupancyGridUpdate
from collections import namedtuple
from Queue import PriorityQueue



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


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def is_same_edge(e1, e2):
    return (e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])


class CleaningBlocks:

    def __init__(self, ms):
        self.triangles = []
        self.triangle_order = []
        self.occ_map = ms.map_arr
        self.map_size = ms.map_arr.shape
        self.rect = (0, 0, self.map_size[1], self.map_size[0])
        self.graph = Graph()

        # find corners
        self.map_rgb = ms.map_rgb
        # map_img_th = thresh.copy()
        # im2, contours, hierarchy = cv.findContours(map_img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(map_img_th, contours, -1, (255, 255, 255), 3)
        corners = cv.goodFeaturesToTrack(ms.map_binary, 20, 0.01, 10)
        self.corners = np.int0(corners)

        # find triangles
        # Create an instance of Subdiv2D
        self.sub_div = cv.Subdiv2D(self.rect)
        self.sub_div = cv.Subdiv2D(self.rect)
        # Insert points into sub_div
        self.sub_div.insert(corners)
        # Filter triangles outside the polygon
        self.extract_triangles()

    def extract_triangles(self):
        initial_triangles = self.sub_div.getTriangleList()

        r = self.rect
        for t in initial_triangles:

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            delaunay_color=(255,0,0)
            img = self.map_rgb

            if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
                center = (np.round((t[0] + t[2] + t[4]) / 3), np.round((t[1] + t[3] + t[5]) / 3))
                center2 = (np.uint32(np.round((t[1] + t[3] + t[5]) / 3)), np.uint32(np.round((t[0] + t[2] + t[4]) / 3)))
                # cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
                # cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)
                # cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
                if self.point_in_room(center2):
                    # if self.line_in_room(mid1) and self.line_in_room(mid2) and self.line_in_room(mid3):
                    mat = np.array([[t[0], t[1], 1], [t[2], t[3], 1], [t[4], t[5], 1]])
                    area = np.linalg.det(mat) / 2

                    tri_edges = [[pt1, pt2, distance(pt1, pt2)], [pt1, pt3, distance(pt1, pt3)],
                                 [pt2, pt3, distance(pt2, pt3)]]
                    # center_edges = [[pt1, center, distance(pt1, center)], [pt1, center, distance(pt1, center)],
                    #                 [pt2, center, distance(pt2, center)]]
                    self.triangles.append(Triangle(t, center, area, tri_edges))
                    last_tri_ind = len(self.triangles)-1
                    self.add_adjacent_tri_edge(last_tri_ind)
        # cv.imshow('delaunay', img)
        # cv.waitKey(0)
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
            # cv.imshow('delaunay', img)
            # cv.waitKey(0)

        # Show results
        cv.imshow('delaunay', img)
        cv.waitKey(0)
        # cv.imwrite('delaunay.jpg', img)

    def point_in_room(self, mid_p_i):    # Funny - tried to check lines
        map = self.occ_map

        if map[mid_p_i] != -1:  # mid pixel
            return True
        return False


    def add_adjacent_tri_edge(self, last_tri_ind):
        for i in range(last_tri_ind):
            if self.is_neighbor(i, last_tri_ind):
                a = self.triangles[i].center
                b = self.triangles[last_tri_ind].center
                self.graph.add_edge(i, last_tri_ind, distance(a, b))

    def is_neighbor(self, v_i, u_i): #Funny
        v_cor = self.triangles[v_i].coordinates
        u_cor = self.triangles[u_i].coordinates
        v_edges = self.triangles[v_i].edges
        u_edges = self.triangles[u_i].edges
        # indices = range(0, 6, 2)
        # for i in indices:
        #     for j in indices:
        #         if v_cor[i] == u_cor[j]:
        #             if v_cor[i+1] == u_cor[j+1]:
        #                 return True
        for e1 in v_edges:
            for e2 in u_edges:
                if is_same_edge(e1, e2):
                    return True
        return False

    def locate_initial_pose(self, first_pose):
        min_dist = 1000
        ind = 0
        x = first_pose[0]
        y = first_pose[1]
        for (i, triangle) in enumerate(self.triangles):
            c = triangle.center
            dist = distance(c, first_pose)
            if dist < min_dist:
                min_dist = dist
                ind = i
        return ind

    def draw_triangle_order(self):
        img = self.map_rgb
        triangle_order=self.triangle_order
        for (i, tri) in enumerate(self.triangles):
            c1 = tri.center
            c1 = tuple(np.uint32((round(c1[0]), round(c1[1]))))

            if i < len(triangle_order)-1:
                c2 = self.triangles[i+1].center
                c2 = tuple(np.uint32((round(c2[0]), round(c2[1]))))
                cv.line(img, c1, c2, (255, 0, i * 7), 1, cv.LINE_AA, 0)

            cv.circle(img, c1, 2, (255, 0, i * 7), cv.FILLED, cv.LINE_AA)

    def sort(self, first_pose):

        starting_point_ind = self.locate_initial_pose(first_pose)
        dist_mat = []
        dict_vector = []
        for i in range(len(self.triangles)):
            dist_vector = self.graph.dijkstra(i)
            dist_mat.append(dist_vector.values())
            dict_vector.append(dist_vector)
            # print(dist_vector)

        triangle_order = [starting_point_ind]
        curr = starting_point_ind
        while len(dict_vector) is not len(triangle_order):
            min_d = np.inf
            next = curr
            for key, dist in dict_vector[curr].items():
                if key is not curr and dist < min_d:
                    min_d = dist
                    next = key
            triangle_order.append(next)
            for key in dict_vector[curr].keys():
                if key is not curr:
                    if curr in dict_vector[key]:
                        dict_vector[key].pop(curr)
            curr = next
        # print(dist_mat)
        # print(triangle_order)
        self.triangle_order = triangle_order
        sorted_triangles = [None] * len(self.triangle_order)
        j = 0
        for i in self.triangle_order:
            sorted_triangles[j] = self.triangles[i]
            j += 1

        self.triangles = sorted_triangles
        return self.triangles




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
        self.initial_pose = None
        rospy.wait_for_service('static_map')
        static_map = rospy.ServiceProxy('static_map', GetMap)
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.init_pose)

        self.map_data = static_map().map
        self.map_org = np.array([self.map_data.info.origin.position.x, self.map_data.info.origin.position.y])
        shape = self.map_data.info.height, self.map_data.info.width
        self.map_arr = np.array(self.map_data.data, dtype='float32').reshape(shape)
        self.resolution = self.map_data.info.resolution
        res,map_binary = cv.threshold(self.map_arr, 90, 255, 0)
        self.map_binary = np.uint8(map_binary)
        self.map_rgb = cv.cvtColor(self.map_binary, cv.COLOR_GRAY2BGR)

    def show_map(self, point=None):
        plt.imshow(self.map_arr)
        if point is not None:
            plt.scatter([point[0]], [point[1]])
        plt.show()

    def position_to_map(self, pos):
        # print("pos", pos)
        # print("self.map_org", self.map_org)
        # print("self.resolution", self.resolution)
        return (pos - self.map_org) // self.resolution

    def map_to_position(self, indices):
        return indices * self.resolution + self.map_org

    def init_pose(self, msg):
        self.initial_pose = msg.pose.pose
        print("initial pose is")
        print("X=" + str(self.initial_pose.position.x))
        print("y=" + str(self.initial_pose.position.y))

    def get_first_pose(self):
        # Waits for YOU to set the initial_pose
        i = 0
        while ms.initial_pose is None:
            if i % 5 == 0:
                print("Waiting for initial_pose. i =", i)
            i += 1
            time.sleep(1.0)
        # print("initial_pose:", ms.initial_pose)

        pos = np.array([self.initial_pose.position.x, self.initial_pose.position.y])
        return self.position_to_map(pos)


# Based on https://stackabuse.com/dijkstras-algorithm-in-python/
class Graph:
    def __init__(self):
        self.edges = {}
        self.visited = []

    def add_edges(self, edges):
        for e in edges:
            self.add_edge(e[0], e[1], e[2])

    def add_edge(self, u, v, weight):
        if u in self.edges:
            self.edges[u].append((v, weight))
        else:
            self.edges[u] = [(v, weight)]

        if v in self.edges:
            self.edges[v].append((u, weight))
        else:
            self.edges[v] = [(u, weight)]

    def dijkstra(self, start_vertex):
        num_vertices = len(self.edges)
        D = {v: float('inf') for v in self.edges.keys()}
        D[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))
        visited = []
        while not pq.empty():
            (dist, current_vertex) = pq.get()
            visited.append(current_vertex)

            for neighbor, dist in self.edges[current_vertex]:
                if neighbor in self.edges:
                    if neighbor not in visited:
                        old_cost = D[neighbor]
                        new_cost = D[current_vertex] + dist
                        if new_cost < old_cost:
                            pq.put((new_cost, neighbor))
                            D[neighbor] = new_cost

        return D


        pos = np.array([self.initial_pose.position.x, self.initial_pose.position.y])
        return self.position_to_map(pos)


class Path_finder:
    def __init__(self, robot_width=5.0, error_gap=0.15, divide_walk_every=20.0):#(10.0 / 3.0)):#robot_width=0.105
        self.error_gap                                 = error_gap
        self.robot_width                               = robot_width
        self.divide_walk_every                         = divide_walk_every
        self.robot_width_with_error_gap                = self.robot_width * (1.0 + max(0.0, self.error_gap))
        self.distance_to_stop_before_next_triangle     = self.robot_width_with_error_gap
        self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap

    def find(self, triangles):
        triangles = self.sort_vertices_by_prev_center_to_closest_next_vertex(triangles)

        path = []
        for triangle in triangles:
            lines = []
            self.add_collision_points_to_lines(lines, triangle[0], triangle[1], triangle[2])
            self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap
            path.append(lines)

        final_borders = []
        final_path    = []
        for i in range(len(path)):
            current_path           = path[i]
            current_path_triangles = []
            for j in range(0, len(current_path), 3):
                x = []
                y = []
                x.append(current_path[j    ][0][0])
                y.append(current_path[j    ][0][1])
                x.append(current_path[j    ][1][0])
                y.append(current_path[j    ][1][1])
                x.append(current_path[j + 1][0][0])
                y.append(current_path[j + 1][0][1])
                x.append(current_path[j + 1][1][0])
                y.append(current_path[j + 1][1][1])
                x.append(current_path[j + 2][0][0])
                y.append(current_path[j + 2][0][1])
                x.append(current_path[j + 2][1][0])
                y.append(current_path[j + 2][1][1])
                current_path_triangles.append((x, y))
                if j == 0:
                    final_borders.append((x, y))

            def add_straight_walk(final_path, current_path_triangle, start_index, distance_decrease_multiplier):
                yaw_angle                = math.atan2(current_path_triangle[1][start_index + 1] - current_path_triangle[1][start_index], current_path_triangle[0][start_index + 1] - current_path_triangle[0][start_index])
                direction_vector         = np.array((current_path_triangle[0][start_index + 1] - current_path_triangle[0][start_index], current_path_triangle[1][start_index + 1] - current_path_triangle[1][start_index], 0))
                direction_vector_len     = np.linalg.norm(direction_vector)
                direction_vector_norm    = direction_vector / direction_vector_len
                direction_vector_new_len = direction_vector_len - (distance_decrease_multiplier * self.distance_to_stop_before_next_triangle)
                direction_vector_new_len = direction_vector_new_len
                number_of_segments       = direction_vector_new_len / self.divide_walk_every
                new_direction_vector     = np.array((current_path_triangle[0][start_index], current_path_triangle[1][start_index], 0))# + direction_vector_norm * self.distance_to_stop_before_and_after_corner#(0 * self.divide_walk_every)
                final_path.append({"position": (new_direction_vector[0], new_direction_vector[1]), "angle": yaw_angle})
                i = 1.0
                while i < number_of_segments:
                    new_direction_vector = np.array((current_path_triangle[0][start_index], current_path_triangle[1][start_index], 0)) + direction_vector_norm * min((i * self.divide_walk_every), direction_vector_new_len)
                    final_path.append({"position": (new_direction_vector[0], new_direction_vector[1]), "angle": yaw_angle})
                    i += 1.0
                new_direction_vector = np.array((current_path_triangle[0][start_index], current_path_triangle[1][start_index], 0)) + direction_vector_norm * direction_vector_new_len
                final_path.append({"position": (new_direction_vector[0], new_direction_vector[1]), "angle": yaw_angle})

            if len(current_path_triangles) == 1:
                current_path_triangle = current_path_triangles[0]
                center_x              = (current_path_triangle[0][0] + current_path_triangle[0][2] + current_path_triangle[0][4]) / 3.0
                center_y              = (current_path_triangle[1][0] + current_path_triangle[1][2] + current_path_triangle[1][4]) / 3.0
                final_path.append({"position": (center_x, center_y), "angle": final_path[-1]["angle"]})
            else:
                for j in range(1, len(current_path_triangles)):
                    current_path_triangle = current_path_triangles[j]
                    yaw_angle             = math.atan2(current_path_triangle[1][1] - current_path_triangle[1][0], current_path_triangle[0][1] - current_path_triangle[0][0])
                    final_path.append({"position": (current_path_triangle[0][0], current_path_triangle[1][0]), "angle": yaw_angle})
                    add_straight_walk(final_path=final_path, current_path_triangle=current_path_triangle, start_index=0, distance_decrease_multiplier=0)
                    add_straight_walk(final_path=final_path, current_path_triangle=current_path_triangle, start_index=2, distance_decrease_multiplier=0)
                    add_straight_walk(final_path=final_path, current_path_triangle=current_path_triangle, start_index=4, distance_decrease_multiplier=1)

        return final_borders, final_path

    def sort_vertices_by_prev_center_to_closest_next_vertex(self, triangles):
        result = []
        result.append(triangles[0])
        for i in range(1, len(triangles)):
            prev_triangle        = triangles[i - 1]
            current_triangle     = triangles[i    ]
            prev_triangle_center = ((prev_triangle[0] + prev_triangle[1] + prev_triangle[2]) / 3.0)
            current_lines        = []
            self.add_collision_points_to_lines(current_lines, current_triangle[0], current_triangle[1], current_triangle[2], only_one_iteration=True)
            if 3 < len(current_lines):
                min_distance       = np.linalg.norm(prev_triangle_center - np.array(current_lines[3][0]))
                min_distance_index = 0
                for j in range(1, 3):
                    distance = np.linalg.norm(prev_triangle_center - np.array(current_lines[3 + j][0]))
                    if distance < min_distance:
                        min_distance       = distance
                        min_distance_index = j
                result.append((current_triangle[min_distance_index % 3], current_triangle[(min_distance_index + 1) % 3], current_triangle[(min_distance_index + 2) % 3]))
            else:
                min_distance       = np.linalg.norm(prev_triangle_center - current_triangle[0])
                min_distance_index = 0
                for j in range(1, 3):
                    distance = np.linalg.norm(prev_triangle_center - current_triangle[j])
                    if distance < min_distance:
                        min_distance       = distance
                        min_distance_index = j
                result.append((current_triangle[min_distance_index % 3], current_triangle[(min_distance_index + 1) % 3], current_triangle[(min_distance_index + 2) % 3]))
        return result

    def add_collision_points_to_lines(self, lines, triangle_point_1, triangle_point_2, triangle_point_3, only_one_iteration=False):
        i     = 0
        first = True
        while True:
            self.add_triangle_to_list(lines, triangle_point_1, triangle_point_2, triangle_point_3)
            if only_one_iteration:
                if i == 1:
                    return
            if first:
                self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap
                # self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap / 2
                first = False
            else:
                self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap
            inner_triangle_point_1 = self.find_inner_triangle_point(triangle_point_1, triangle_point_3, triangle_point_2)
            if inner_triangle_point_1 is None:
                return
            inner_triangle_point_2 = self.find_inner_triangle_point(triangle_point_2, triangle_point_1, triangle_point_3)
            if inner_triangle_point_2 is None:
                return
            inner_triangle_point_3 = self.find_inner_triangle_point(triangle_point_3, triangle_point_2, triangle_point_1)
            if inner_triangle_point_3 is None:
                return
            triangle_point_1, triangle_point_2, triangle_point_3 = inner_triangle_point_1, inner_triangle_point_2, inner_triangle_point_3
            i += 1

    def add_triangle_to_list(self, lines, triangle_point_1, triangle_point_2, triangle_point_3):
        self.add_line_to_list(lines, ((triangle_point_1[0], triangle_point_1[1], triangle_point_1[2]),
                                      (triangle_point_2[0], triangle_point_2[1], triangle_point_2[2])))
        self.add_line_to_list(lines, ((triangle_point_2[0], triangle_point_2[1], triangle_point_2[2]),
                                      (triangle_point_3[0], triangle_point_3[1], triangle_point_3[2])))
        self.add_line_to_list(lines, ((triangle_point_3[0], triangle_point_3[1], triangle_point_3[2]),
                                      (triangle_point_1[0], triangle_point_1[1], triangle_point_1[2])))

    def add_line_to_list(self, lines, line_to_add):
        if (line_to_add[1], line_to_add[0]) not in lines:
            lines.append(line_to_add)

    def find_inner_triangle_point(self, triangle_point, other_triangle_point_1, other_triangle_point_2):
        triangle_center              = (triangle_point + other_triangle_point_1 + other_triangle_point_2) / 3
        line_1                       = other_triangle_point_1 - triangle_point
        line_2                       = other_triangle_point_2 - triangle_point
        line_3                       = other_triangle_point_2 - other_triangle_point_1
        cross                        = np.cross(line_1, line_2)
        orthogonal_line_1            = np.cross(cross, line_1)
        orthogonal_line_2            = np.cross(cross, line_2)
        orthogonal_line_1_norm       = orthogonal_line_1 / np.linalg.norm(orthogonal_line_1)
        orthogonal_line_2_norm       = orthogonal_line_2 / np.linalg.norm(orthogonal_line_2)
        orthogonal_line_1_final_size = self.margin_between_outter_and_inner_triangles * orthogonal_line_1_norm
        orthogonal_line_2_final_size = self.margin_between_outter_and_inner_triangles * orthogonal_line_2_norm
        line_1_start                 = triangle_point + orthogonal_line_1_final_size
        line_1_end                   = line_1 / np.linalg.norm(line_1)
        line_2_start                 = triangle_point + orthogonal_line_2_final_size
        line_2_end                   = line_2 / np.linalg.norm(line_2)
        line_3_start                 = triangle_point - orthogonal_line_1_final_size
        line_3_end                   = line_1 / np.linalg.norm(line_1)
        line_4_start                 = triangle_point - orthogonal_line_2_final_size
        line_4_end                   = line_2 / np.linalg.norm(line_2)
        possible_point_1 = self.nearest_intersection(points=np.array([[line_1_start[0], line_1_start[1], line_1_start[2]], [line_2_start[0], line_2_start[1], line_2_start[2]]]),
                                                     dirs=np.array([[line_1_end[0], line_1_end[1], line_1_end[2]], [line_2_end[0], line_2_end[1], line_2_end[2]]]))
        possible_point_2 = self.nearest_intersection(points=np.array([[line_1_start[0], line_1_start[1], line_1_start[2]], [line_4_start[0], line_4_start[1], line_4_start[2]]]),
                                                     dirs=np.array([[line_1_end[0], line_1_end[1], line_1_end[2]], [line_4_end[0], line_4_end[1], line_4_end[2]]]))
        possible_point_3 = self.nearest_intersection(points=np.array([[line_3_start[0], line_3_start[1], line_3_start[2]], [line_2_start[0], line_2_start[1], line_2_start[2]]]),
                                                     dirs=np.array([[line_3_end[0], line_3_end[1], line_3_end[2]], [line_2_end[0], line_2_end[1], line_2_end[2]]]))
        possible_point_4 = self.nearest_intersection(points=np.array([[line_3_start[0], line_3_start[1], line_3_start[2]], [line_4_start[0], line_4_start[1], line_4_start[2]]]),
                                                     dirs=np.array([[line_3_end[0], line_3_end[1], line_3_end[2]], [line_4_end[0], line_4_end[1], line_4_end[2]]]))
        possible_point_1_to_center_distance = np.linalg.norm(possible_point_1 - triangle_center)
        possible_point_2_to_center_distance = np.linalg.norm(possible_point_2 - triangle_center)
        possible_point_3_to_center_distance = np.linalg.norm(possible_point_3 - triangle_center)
        possible_point_4_to_center_distance = np.linalg.norm(possible_point_4 - triangle_center)
        closest_point_to_center             = possible_point_1
        closest_point_to_center_distance    = possible_point_1_to_center_distance
        if possible_point_2_to_center_distance < closest_point_to_center_distance:
            closest_point_to_center          = possible_point_2
            closest_point_to_center_distance = possible_point_2_to_center_distance
        if possible_point_3_to_center_distance < closest_point_to_center_distance:
            closest_point_to_center          = possible_point_3
            closest_point_to_center_distance = possible_point_3_to_center_distance
        if possible_point_4_to_center_distance < closest_point_to_center_distance:
            closest_point_to_center          = possible_point_4
        A                              = other_triangle_point_1
        AB                             = line_3
        AP                             = closest_point_to_center - other_triangle_point_1
        point_on_line_projection_point = A + np.dot(AP, AB) / np.dot(AB, AB) * AB
        if self.margin_between_outter_and_inner_triangles <= np.linalg.norm(closest_point_to_center - point_on_line_projection_point) and \
            np.linalg.norm(closest_point_to_center - triangle_point) < np.linalg.norm(point_on_line_projection_point - triangle_point):
                return closest_point_to_center
        return None

    def nearest_intersection(self, points, dirs):
        """
        :param points: (N, 3) array of points on the lines
        :param dirs: (N, 3) array of unit direction vectors
        :returns: (3,) array of intersection point
        """
        dirs_mat   = np.matmul(dirs[:, :, np.newaxis], dirs[:, np.newaxis, :])
        # dirs_mat   = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
        points_mat = points[:, :, np.newaxis]
        I          = np.eye(3)
        return np.linalg.lstsq(
            (I - dirs_mat).sum(axis=0),
            (np.matmul((I - dirs_mat), points_mat)).sum(axis=0),
            # ((I - dirs_mat) @ points_mat).sum(axis=0),
            rcond=None
        )[0][:, 0]

def array_to_quaternion(nparr):
    '''
    Takes a numpy array holding the members of a quaternion and returns it as a
    geometry_msgs.msg.Quaternion message instance.
    '''
    quat = Quaternion()
    quat.x = nparr[0]
    quat.y = nparr[1]
    quat.z = nparr[2]
    quat.w = nparr[3]
    return quat

def movebase_client(map_service, path):
    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

    for current_goal in path:
        # Creates a MoveBaseGoal object
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"

        # Calculates the goal's pose
        # Calculates the goal's position
        position = map_service.map_to_position(indices=np.array((current_goal["position"][0], current_goal["position"][1])))
        goal.target_pose.pose.position.x = position[0]
        goal.target_pose.pose.position.y = position[1]
        # Calculates the goal's angle
        quaternionArray = tf.transformations.quaternion_about_axis(current_goal["angle"], (0, 0, 1))
        goal.target_pose.pose.orientation = array_to_quaternion(quaternionArray)

        # Saves the current time
        goal.target_pose.header.stamp = rospy.Time.now()

        # Sends the goal to the action server.
        rospy.loginfo("Sending goal: x: {x}, y: {y}, angle: {angle}.".format(x=round(goal.target_pose.pose.position.x, 2), y=round(goal.target_pose.pose.position.y, 2), angle=round(goal.target_pose.pose.orientation.w, 2)))
        client.send_goal(goal)
        rospy.loginfo("New goal command received!")

        # Waits for the server to finish performing the action.
        wait = client.wait_for_result()

        # If the result doesn't arrive, assume the server is not available
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            # Result of executing the action
            rospy.loginfo("Current goal has been reached!")

    return client.get_result()

def plot_path(borders, path, plot, save_to_file):
    if plot or save_to_file:
        x = []
        y = []
        for i in range(len(path) - 1):
            x.append(path[i    ]["position"][0])
            y.append(path[i    ]["position"][1])
            x.append(path[i + 1]["position"][0])
            y.append(path[i + 1]["position"][1])
            angle_radian    = path[i]["angle"]
            rotation_matrix = R.from_euler('z', angle_radian, degrees=False)
            rotated_vector  = rotation_matrix.apply(np.array((0.05, 0.0, 0.0)))
            plt.arrow(x=path[i]["position"][0], y=path[i]["position"][1], dx=rotated_vector[0], dy=rotated_vector[1], width=0.5)#.015)
        for i in range(len(borders)):
            plt.plot(np.array(borders[i][0]), np.array(borders[i][1]))
        plt.plot(np.array(x), np.array(y))
        plt.annotate(
            'Start', xy=(x[0], y[0]), xytext=(x[0], y[0] - 0.5),
            horizontalalignment="center",
            arrowprops=dict(arrowstyle='->', lw=1)
        )
        plt.annotate(
            'End', xy=(x[-1], y[-1]), xytext=(x[-1], y[-1] + 0.75),
            horizontalalignment="center",
            arrowprops=dict(arrowstyle='->', lw=1)
        )
        plt.axis('scaled')
        if save_to_file:
            plt.savefig("path.png")
        if plot:
            plt.show()

def move_robot_on_path(map_service, path):
    try:
       # Initializes a rospy node to let the SimpleActionClient publish and subscribe
       #  rospy.init_node('movebase_client_py')
        result = movebase_client(map_service=map_service, path=path)
        if result:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation Exception.")


def vacuum_cleaning(ms, cb):

    print('start vacuum_cleaning')
    triangle_list = cb.get_triangles()

    # Draw delaunay triangles
    cb.draw_triangle_order()
    cb.draw_triangles((0, 255, 0))
    triangles = []  # Tom's format
    for triangle in triangle_list:
        t = triangle.coordinates
        triangles.append((np.array((t[0], t[1], 0)), np.array((t[4], t[5], 0)), np.array((t[2], t[3], 0))))

    # Path planning
    path_finder = Path_finder()
    borders, path = path_finder.find(triangles=triangles)
    print("Done creating the path. Length:", len(path))

    # Plots / Saves the path map
    plot_path(borders=borders, path=path, plot=False, save_to_file=True)

    # Moves the robot according to the path
    move_robot_on_path(map_service=ms, path=path)

# class MoveBaseSeq():
#
#     def __init__(self):
#
#         rospy.init_node('move_base_sequence')
#         points_seq = rospy.get_param('move_base_seq/p_seq')
#         # Only yaw angle required (no ratotions around x and y axes) in deg:
#         yaweulerangles_seq = rospy.get_param('move_base_seq/yea_seq')
#         # List of goal quaternions:
#         quat_seq = list()
#         # List of goal poses:
#         self.pose_seq = list()
#         self.goal_cnt = 0
#         for yawangle in yaweulerangles_seq:
#             # Unpacking the quaternion list and passing it as arguments to Quaternion message constructor
#             quat_seq.append(Quaternion(*(quaternion_from_euler(0, 0, yawangle * math.pi / 180, axes='sxyz'))))
#         n = 3
#         # Returns a list of lists [[point1], [point2],...[pointn]]
#         points = [points_seq[i:i + n] for i in range(0, len(points_seq), n)]
#         for point in points:
#             # Exploit n variable to cycle in quat_seq
#             self.pose_seq.append(Pose(Point(*point), quat_seq[n - 3]))
#             n += 1
#         # Create action client
#         self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
#         rospy.loginfo("Waiting for move_base action server...")
#         wait = self.client.wait_for_server(rospy.Duration(5.0))
#         if not wait:
#             rospy.logerr("Action server not available!")
#             rospy.signal_shutdown("Action server not available!")
#             return
#         rospy.loginfo("Connected to move base server")
#         rospy.loginfo("Starting goals achievements ...")
#         self.movebase_client()
#
#     def active_cb(self):
#         rospy.loginfo("Goal pose " + str(self.goal_cnt + 1) + " is now being processed by the Action Server...")
#
#     def feedback_cb(self, feedback):
#         # To print current pose at each feedback:
#         # rospy.loginfo("Feedback for goal "+str(self.goal_cnt)+": "+str(feedback))
#         rospy.loginfo("Feedback for goal pose " + str(self.goal_cnt + 1) + " received")
#
#     def done_cb(self, status, result):
#         self.goal_cnt += 1
#         # Reference for terminal status values: http://docs.ros.org/diamondback/api/actionlib_msgs/html/msg/GoalStatus.html
#         if status == 2:
#             rospy.loginfo("Goal pose " + str(
#                 self.goal_cnt) + " received a cancel request after it started executing, completed execution!")
#
#         if status == 3:
#             rospy.loginfo("Goal pose " + str(self.goal_cnt) + " reached")
#             if self.goal_cnt < len(self.pose_seq):
#                 next_goal = MoveBaseGoal()
#                 next_goal.target_pose.header.frame_id = "map"
#                 next_goal.target_pose.header.stamp = rospy.Time.now()
#                 next_goal.target_pose.pose = self.pose_seq[self.goal_cnt]
#                 rospy.loginfo("Sending goal pose " + str(self.goal_cnt + 1) + " to Action Server")
#                 rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
#                 self.client.send_goal(next_goal, self.done_cb, self.active_cb, self.feedback_cb)
#             else:
#                 rospy.loginfo("Final goal pose reached!")
#                 rospy.signal_shutdown("Final goal pose reached!")
#                 return
#
#         if status == 4:
#             rospy.loginfo("Goal pose " + str(self.goal_cnt) + " was aborted by the Action Server")
#             rospy.signal_shutdown("Goal pose " + str(self.goal_cnt) + " aborted, shutting down!")
#             return
#
#         if status == 5:
#             rospy.loginfo("Goal pose " + str(self.goal_cnt) + " has been rejected by the Action Server")
#             rospy.signal_shutdown("Goal pose " + str(self.goal_cnt) + " rejected, shutting down!")
#             return
#
#         if status == 8:
#             rospy.loginfo("Goal pose " + str(
#                 self.goal_cnt) + " received a cancel request before it started executing, successfully cancelled!")
#
#     def movebase_client_next_goal(self):
#         goal = MoveBaseGoal()
#         goal.target_pose.header.frame_id = "map"
#         goal.target_pose.header.stamp = rospy.Time.now()
#         goal.target_pose.pose = self.pose_seq[self.goal_cnt]
#         rospy.loginfo("Sending goal pose " + str(self.goal_cnt + 1) + " to Action Server")
#         rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
#         self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)
#         rospy.spin()
#
#     ### INSPECTION ###

class InspectionCostmapUpdater:
    def __init__(self, ms):
        self.last_msg = None
        self.ms = ms
        self.differences_map_file = 'differences_map.png'
        self.occ_map = self.binary_dilation(map=ms.map_arr, iterations1=0, iterations2=1)

        map_dilated = cv.dilate(ms.map_arr, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))
        res, map_dilated = cv.threshold(map_dilated, 95, 255, 0)
        self.occ_map_dilated = np.uint8(map_dilated)

        self.cost_map = None
        self.differences_map = None
        self.shape = None
        rospy.Subscriber('/move_base/global_costmap/costmap'        , OccupancyGrid      , self.init_costmap_callback  )
        rospy.Subscriber('/move_base/global_costmap/costmap_updates', OccupancyGridUpdate, self.costmap_callback_update)
        self.ins_pub = rospy.Publisher('/inspection_report', std_msgs.msg.String, queue_size=10)

    def binary_dilation(self,map, iterations1, iterations2):
        occ_map_ = self.map_to_binary_map(map=map)
        if 0 < iterations1:
            struct1 = ndimage.generate_binary_structure(2, 2)
            occ_map_ = ndimage.binary_dilation(occ_map_, structure=struct1, iterations=iterations1).astype(occ_map_.dtype)
        if 0 < iterations2:
            struct2 = ndimage.generate_binary_structure(2, 1)
            occ_map_ = ndimage.binary_dilation(occ_map_, structure=struct2, iterations=iterations2).astype(occ_map_.dtype)
        return occ_map_

    def map_to_binary_map(self, map):
        map_ = np.zeros(shape=map.shape)
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if (map[i][j] == 100.0) or (map[i][j] == 1.0):
                    map_[i][j] = 1.0
                else:
                    map_[i][j] = 0.0
        return map_

    def init_costmap_callback(self, msg):
        self.shape = msg.info.height, msg.info.width
        self.cost_map = np.array(msg.data, dtype=np.uint8).reshape(self.shape)
        self.last_msg = msg

    def costmap_callback_update(self, msg):
        if msg.header.stamp.secs - self.last_msg.header.stamp.secs > 7:
            shape = msg.height, msg.width
            data = np.array(msg.data, dtype=np.uint8).reshape(shape)
            #clear contour before counting
            self.cost_map[msg.y:msg.y + shape[0], msg.x: msg.x + shape[1]] = data

            # cost_map_ = np.where(self.cost_map < 90, 0, self.cost_map)
            res, cost_binary = cv.threshold(self.cost_map, 95, 255, 0)
            # cv.imwrite('figs/cost_map' + str(self.last_msg.header.stamp.secs) + '.jpg', self.cost_map)
            # cv.imshow('cost_map', self.cost_map)
            # cv.waitKey(0)
            # cv.imwrite('figs/map_binary' + str(self.last_msg.header.stamp.secs) + '.jpg', cost_binary)
            # cv.imshow('map_binary', map_binary)
            # cv.waitKey(0)
            differences_map = np.abs(cost_binary-self.occ_map_dilated)
            cv.imwrite('figs/differences_map' + str(self.last_msg.header.stamp.secs) + '.jpg', differences_map)
            # cv.imshow('differences_map', differences_map)
            # cv.waitKey(0)
            rows = cost_binary.shape[0]
            circles = cv.HoughCircles(differences_map, cv.HOUGH_GRADIENT, 1, rows / 16,  # / 8
                                      param1=100, param2=9,  # param2 - accumulate centers - the smaller it is the more false circles
                                      minRadius=4, maxRadius=11)

            num_of_circles = 0
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    # circle center
                    cv.circle(cost_binary, center, 1, (0, 100, 100), 3)
                    # circle outline
                    radius = i[2]
                    cv.circle(cost_binary, center, radius, (255, 0, 255), 3)
                num_of_circles = circles.shape[1]
                cv.imwrite("figs/detected circles_" + str(num_of_circles) + '_' + str(self.last_msg.header.stamp.secs) + '.jpg', cost_binary)
                # cv.imshow("detected circles_" + str(len(circles)), cost_binary)
                # cv.waitKey(0)

            print("detected circles ", num_of_circles)
            self.ins_pub.publish(std_msgs.msg.String(str(num_of_circles) + " spheres detected at time " + str(msg.header.stamp)))
            print(str(num_of_circles) + " spheres detected at time " + str(msg.header.stamp))
            self.last_msg = msg

    # def costmap_callback_update(self, msg):
    #     if(msg.header.stamp.secs-self.last_msg.header.stamp.secs > 5):
    #         self.last_msg = msg
    #         shape = msg.height, msg.width
    #         data = np.array(msg.data).reshape(shape)
    #         self.cost_map[msg.y:msg.y + shape[0], msg.x: msg.x + shape[1]] = data
    #
    #         cost_map_ = np.where(self.cost_map < 90, 0, self.cost_map)
    #         plt.imshow(cost_map_)
    #         plt.show()
    #         cost_map_ = self.map_to_binary_map(map=cost_map_)
    #         # exit(-1)
    #         self.differences_map = ms.map_arr - self.occ_map
    #         self.differences_map = np.where(self.differences_map < 0.0, 0.0, self.differences_map)
    #         self.differences_map = self.binary_dilation(map=self.differences_map, iterations1=3, iterations2=2)
    #         plt.imshow(self.differences_map)
    #         # plt.show()
    #         plt.savefig(self.differences_map_file)
    #         num_of_circles = self.calculate_number_of_circles_in_map()
    #         self.ins_pub.publish(std_msgs.msg.String(str(num_of_circles) +"spheres detected at time" + str(msg.header.stamp)))
    #         print(str(num_of_circles) + "spheres detected at time " + str(msg.header.stamp))
    #
    #         self.show_map()

    # def calculate_number_of_circles_in_map(self):
    #     filename = self.differences_map_file
    #     # Loads an image
    #     src      = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    #     # Check if image is loaded fine
    #     if src is None:
    #         print ('Error opening image!')
    #         print ('Usage: hough_circle.py [image_name -- default ' + self.differences_map_file + '] \n')
    #         return -1
    #     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #     # gray = cv.medianBlur(gray, 5)
    #     plt.imshow(gray)
    #     plt.show()
    #     rows = gray.shape[0]
    #     circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1,
    #                               mindist=5,#rows / 16,#/ 8
    #                               param1=100, param2=15,
    #                               # param1=100, param2=30,
    #                               minRadius=5, maxRadius=11)
    #                               # minRadius=1, maxRadius=30)
    #     if circles is not None:
    #         circles = np.uint16(np.around(circles))
    #         for i in circles[0, :]:
    #             center = (i[0], i[1])
    #             # circle center
    #             cv.circle(src, center, 1, (0, 100, 100), 3)
    #             # circle outline
    #             radius = i[2]
    #             cv.circle(src, center, radius, (255, 0, 255), 3)
    #         print("detected circles", len(circles))
    #         # cv.imshow("detected circles", src)
    #         cv.imshow("detected circles" + str(len(circles)), src)
    #         cv.waitKey(0)
    #
    #     print("detected 0 circles")
    #     return 0

    def show_map(self):
        if not self.cost_map is None:
            plt.imshow(self.differences_map)
            # plt.imshow(self.cost_map)
            plt.show()


def inspection(ms, cb):
    print('start inspection')

    occ_map = ms.map_arr
    # cb = CleaningBlocks(occ_map)
    # plt.imshow(occ_map)
    # plt.show()

    cmu = InspectionCostmapUpdater(ms)
    # rospy.spin()

    path = []
    triangles = cb.get_triangles()
    for tri in triangles:
        p=tri.center
        path.append({"position": p, "angle": 0})
    # path.append({"position": (215, 200), "angle": 0})
    move_robot_on_path(map_service=ms, path=path)
    rospy.spin()



if __name__ == '__main__':


    rospy.init_node('get_map_example', anonymous=True)
    reload(logging)

    # rc_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
    # rc_DWA_client.update_configuration({"max_vel_x": 2.5})
    # rc_DWA_client.update_configuration({"max_vel_trans": 2.5})

    ms = MapService()
    # rc_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
    # rc_DWA_client.update_configuration({"max_vel_x": 2.5})
    Triangle = namedtuple('Triangle', ['coordinates', 'center', 'area', 'edges'])
    cb = CleaningBlocks(ms)
    first_pose = ms.get_first_pose()
    cb.sort(first_pose)

    exec_mode = sys.argv[1]

    # RRRRRRRRRRRRRREMOVEEEEEEEEEEEEEEEEEE
    # RRRRRRRRRRRRRREMOVEEEEEEEEEEEEEEEEEE

    # exec_mode = 'inspection'
    # RRRRRRRRRRRRRREMOVEEEEEEEEEEEEEEEEEE
    # RRRRRRRRRRRRRREMOVEEEEEEEEEEEEEEEEEE

    print('exec_mode:' + exec_mode)
    if exec_mode == 'cleaning':
        vacuum_cleaning(ms, cb)
    elif exec_mode == 'inspection':
        inspection(ms, cb)
    else:
        print("Code not found")
        raise NotImplementedError


