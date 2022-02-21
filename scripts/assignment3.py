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
# import tf
import logging
import actionlib
import multi_move_base

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
import dynamic_reconfigure.client
import std_msgs
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from collections import namedtuple
from Queue import PriorityQueue


class CleaningBlocks:

    def __init__(self, ms):
        self.sub_div = None
        self.corners = None
        self.triangles = []
        # self.triangle_order = []
        self.occ_map = ms.map_arr
        self.map_size = ms.map_arr.shape
        self.rect = (0, 0, self.map_size[1], self.map_size[0])
        self.graph = Graph()

        # find corners
        self.map_rgb = ms.map_rgb
        # map_img_th = thresh.copy()

        # find triangles
        self.find_corners()
        # Filter triangles outside the polygon
        self.extract_triangles()

        self.dist_mat = self.graph.get_dist_mat()

    def find_corners(self):
        # im2, contours, hierarchy = cv.findContours(map_img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(map_img_th, contours, -1, (255, 255, 255), 3)

        corners = cv.goodFeaturesToTrack(ms.map_binary, 20, 0.01, 10)
        self.corners = np.int0(corners)

        # Create an instance of Subdiv2D
        self.sub_div = cv.Subdiv2D(self.rect)
        # Insert points into sub_div
        self.sub_div.insert(corners)

    def extract_triangles(self):
        initial_triangles = self.sub_div.getTriangleList()

        r = self.rect
        for t in initial_triangles:

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            delaunay_color = (255, 0, 0)
            img = self.map_rgb

            if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
                center = (np.round((t[0] + t[2] + t[4]) / 3), np.round((t[1] + t[3] + t[5]) / 3))
                center2 = (np.uint32(np.round((t[1] + t[3] + t[5]) / 3)), np.uint32(np.round((t[0] + t[2] + t[4]) / 3)))
                # cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
                # cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)
                # cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
                mid1 = (np.uint32((t[1] + t[3]) / 2), np.uint32((t[0] + t[2]) / 2))
                mid2 = (np.uint32((t[3] + t[5]) / 2), np.uint32((t[2] + t[4]) / 2))
                mid3 = (np.uint32((t[1] + t[5]) / 2), np.uint32((t[0] + t[4]) / 2))
                if self.point_in_room(center2) and (self.line_in_room(mid1) and self.line_in_room(mid2) and
                                                    self.line_in_room(mid3)):
                    mat = np.array([[t[0], t[1], 1], [t[2], t[3], 1], [t[4], t[5], 1]])
                    area = np.linalg.det(mat) / 2

                    tri_edges = [[pt1, pt2, distance(pt1, pt2)], [pt1, pt3, distance(pt1, pt3)],
                                 [pt2, pt3, distance(pt2, pt3)]]
                    # center_edges = [[pt1, center, distance(pt1, center)], [pt1, center, distance(pt1, center)],
                    #                 [pt2, center, distance(pt2, center)]]
                    new_triangle = Triangle(t, center, area, tri_edges)
                    self.triangles.append(new_triangle)
                    last_tri_ind = len(self.triangles) - 1
                    self.add_adjacent_tri_edge(last_tri_ind)
        # cv.imshow('delaunay', img)
        # cv.waitKey(0)

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

    # Draw delaunay triangles
    def draw_triangles(self, delaunay_color, triangles):
        img = self.map_rgb
        # Draw points
        for p in self.corners:
            draw_point(img, p[0], (0, 0, 255))

        for triangle in triangles:
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

    def point_in_room(self, mid_p_i):
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

    def is_neighbor(self, v_i, u_i):
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

    def draw_path(self, triangle_order):
        img = self.map_rgb
        c1 = triangle_order[0].center
        c1 = tuple(np.uint32((round(c1[0]), round(c1[1]))))
        for i in range(1, len(triangle_order)):
            c2 = triangle_order[i].center
            c2 = tuple(np.uint32((round(c2[0]), round(c2[1]))))

            cv.line(img, c1, c2, (255, 0, i * 9), 1, cv.LINE_AA, 0)
            cv.circle(img, c1, 2, (255, 0, i * 9), cv.FILLED, cv.LINE_AA)
            c1 = c2

    def plan_path(self, first_pose, dist_mat, triangles):

        first_ind = self.locate_initial_pose(first_pose)
        # dist_mat = self.dist_mat
        # dict_vector = []
        # for i in range(len(self.triangles)):
        #     dist_vector = self.graph.dijkstra(i)
        #     dist_mat.append(dist_vector.values())
        #     dict_vector.append(dist_vector)
        # print(dist_vector)

        triangle_order = [triangles[first_ind]]
        visited = [first_ind]
        curr = first_ind
        next_tri = None
        num_of_tri = len(triangles)

        for i in range(num_of_tri - 1):
            min_d = np.inf
            for key, dist in dist_mat[curr].items():
                if key not in visited and dist < min_d:
                    min_d = dist
                    next_tri = key
            visited.append(next_tri)
            triangle_order.append(triangles[next_tri])
            curr = next_tri

        # while len(dict_vector) is not len(triangle_order):
        #     min_d = np.inf
        #     next = curr
        #     for key, dist in dict_vector[curr].items():
        #         if key is not curr and dist < min_d:
        #             min_d = dist
        #             next = key
        #     triangle_order.append(next)
        #     for key in dict_vector[curr].keys():
        #         if key is not curr:
        #             if curr in dict_vector[key]:
        #                 dict_vector[key].pop(curr)
        #     curr = next
        # # print(dist_mat)
        # # print(triangle_order)
        # self.triangle_order = triangle_order
        # sorted_triangles = [None] * len(self.triangle_order)
        # j = 0
        # for i in self.triangle_order:
        #     sorted_triangles[j] = self.triangles[i]
        #     j += 1
        #
        # # self.triangles = sorted_triangles
        return triangle_order


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
#     def movebase_client(self):
#         goal = MoveBaseGoal()
#         goal.target_pose.header.frame_id = "map"
#         goal.target_pose.header.stamp = rospy.Time.now()
#         goal.target_pose.pose = self.pose_seq[self.goal_cnt]
#         rospy.loginfo("Sending goal pose " + str(self.goal_cnt + 1) + " to Action Server")
#         rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
#         self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)
#         rospy.spin()


class MapService(object):

    def __init__(self):
        self.initial_pose = None
        ag=0
        rospy.wait_for_service('tb3_%d/static_map'%ag)
        static_map = rospy.ServiceProxy('tb3_%d/static_map'%ag, GetMap)
        rospy.Subscriber('tb3_%d/initialpose'%ag, PoseWithCovarianceStamped, self.init_pose)

        self.map_data = static_map().map
        self.map_org = np.array([self.map_data.info.origin.position.x, self.map_data.info.origin.position.y])
        shape = self.map_data.info.height, self.map_data.info.width
        self.map_arr = np.array(self.map_data.data, dtype='float32').reshape(shape)
        self.resolution = self.map_data.info.resolution
        res, map_binary = cv.threshold(self.map_arr, 90, 255, 0)
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
        while self.initial_pose is None:
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

    def get_num_of_verices(self):
        return len(self.edges)

    def get_vertices(self):
        return self.edges.keys()

    def add_vertex(self, u):
        if u not in self.edges:
            self.edges[u] = []

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

    def get_dist_mat(self):
        num_of_tri = self.get_num_of_verices()
        dist_mat = [None]*num_of_tri
        for v in self.edges:
            dist_mat[v]=(self.dijkstra(v).values())

        # pos = np.array([self.initial_pose.position.x, self.initial_pose.position.y])
        # return self.position_to_map(pos)
        return dist_mat


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


def vacuum_cleaning(agent_id, agent_max_vel):
    x = 0
    y = 1
    print('cleaning (%d,%d)' % (x, y))
    result = multi_move_base.move(agent_id, x, y)

    print('moving agent %d' % agent_id)
    x = 1
    y = 0
    print('cleaning (%d,%d)' % (x, y))
    result = multi_move_base.move(agent_id, x, y)


def inspection(agent_id, agent_max_vel):
    print('start inspection')


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':

    # Initializes a rospy node to let the SimpleActionClient publish and subscribe
    rospy.init_node('assignment3')
    # rospy.init_node('get_map_example', anonymous=True)

    ms = MapService()
    # rc_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
    # rc_DWA_client.update_configuration({"max_vel_x": 2.5})
    Triangle = namedtuple('Triangle', ['coordinates', 'center', 'area', 'edges'])

    # first_pose = ms.get_first_pose()
    first_pose = PoseWithCovarianceStamped()
    first_pose.header.frame_id = "map"
    first_pose.header.stamp = rospy.Time.now()
    first_pose.pose.pose.position.x = 1.0
    # robot2?
    # cb.sort(first_pose)

    exec_mode = sys.argv[1]
    print('exec_mode:' + exec_mode)

    agent_max_vel = 0.22
    if exec_mode == 'cleaning':
        agent_id = sys.argv[2]
        agent_max_vel = sys.argv[3]
        vacuum_cleaning(agent_id, agent_max_vel)

    elif exec_mode == 'inspection':
        agent_id = sys.argv[2]
        agent_max_vel = sys.argv[3]
        cb = CleaningBlocks(ms)
        dist_mat = np.array(cb.dist_mat)
        extreme_nodes = np.where(np.max(dist_mat) == dist_mat)
        # 1. assign nodes to each extreme node
        # 2. plan path to each set
        # 3. transmit the path to the other robot
        if agent_id == '0':
            path = cb.plan_path([first_pose.pose.pose.position.x, first_pose.pose.pose.position.y], cb.dist_mat,
                                cb.triangles)
            cb.draw_path(path)
            cb.draw_triangles((0, 255, 0), cb.triangles)

        inspection(agent_id, agent_max_vel)
    else:
        print("Code not found")
        raise NotImplementedError
