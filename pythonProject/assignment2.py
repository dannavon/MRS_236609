#!/usr/bin/env python2.7

import rospy
import actionlib
import sys
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2 as cv

import matplotlib.pyplot as plt

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid, Odometry
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
        print(pos)
        print(self.map_org)
        print(self.resolution)
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

class Path_finder:
    def __init__(self):
        self.error_gap                                 = 0.1
        self.robot_width                               = 3.0#0.105
        self.divide_walk_every                         = (10.0 / 3.0)#(1.0 / 3.0)
        self.robot_width_with_error_gap             = self.robot_width * (1.0 + max(0.0, self.error_gap))
        self.distance_to_stop_before_and_after_corner  = self.robot_width_with_error_gap
        self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap

    def find(self, triangles):
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
                x.append(current_path[j][0][0])
                y.append(current_path[j][0][1])
                x.append(current_path[j][1][0])
                y.append(current_path[j][1][1])
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
                direction_vector_new_len = direction_vector_len - (distance_decrease_multiplier * self.distance_to_stop_before_and_after_corner)
                direction_vector_new_len = max(direction_vector_new_len, self.distance_to_stop_before_and_after_corner)
                number_of_segments       = direction_vector_new_len / self.divide_walk_every
                i                        = 1.0
                while i < number_of_segments:
                    new_direction_vector = np.array((current_path_triangle[0][start_index], current_path_triangle[1][start_index], 0)) + direction_vector_norm * (i * self.divide_walk_every)
                    final_path.append({"position": (new_direction_vector[0], new_direction_vector[1]), "angle": yaw_angle})
                    i += 1.0
                new_direction_vector = np.array((current_path_triangle[0][start_index], current_path_triangle[1][start_index], 0)) + direction_vector_norm * direction_vector_new_len
                final_path.append({"position": (new_direction_vector[0], new_direction_vector[1]), "angle": yaw_angle})

            def add_corner_turn(final_path, current_path_triangle, start_index):
                yaw_angle                = math.atan2(current_path_triangle[1][start_index + 3] - current_path_triangle[1][start_index + 2], current_path_triangle[0][start_index + 3] - current_path_triangle[0][start_index + 2])
                direction_vector         = np.array((current_path_triangle[0][start_index + 3] - current_path_triangle[0][start_index + 2], current_path_triangle[1][start_index + 3] - current_path_triangle[1][start_index + 2], 0))
                direction_vector_len     = np.linalg.norm(direction_vector)
                direction_vector_norm    = direction_vector / direction_vector_len
                direction_vector_new_len = self.distance_to_stop_before_and_after_corner
                new_direction_vector     = np.array((current_path_triangle[0][start_index + 2], current_path_triangle[1][start_index + 2], 0)) + direction_vector_norm * direction_vector_new_len
                final_path.append({"position": (new_direction_vector[0], new_direction_vector[1]), "angle": yaw_angle})

            def add_straight_walk_and_corner_turn(final_path, current_path_triangle, start_index):
                add_straight_walk(final_path=final_path, current_path_triangle=current_path_triangle, start_index=start_index, distance_decrease_multiplier=1)
                add_corner_turn(final_path=final_path, current_path_triangle=current_path_triangle, start_index=start_index)

            for j in range(1, len(current_path_triangles)):
                current_path_triangle = current_path_triangles[j]
                yaw_angle             = math.atan2(current_path_triangle[1][1] - current_path_triangle[1][0], current_path_triangle[0][1] - current_path_triangle[0][0])
                final_path.append({"position": (current_path_triangle[0][0], current_path_triangle[1][0]), "angle": yaw_angle})
                add_straight_walk_and_corner_turn(final_path=final_path, current_path_triangle=current_path_triangle, start_index=0)
                add_straight_walk_and_corner_turn(final_path=final_path, current_path_triangle=current_path_triangle, start_index=2)
                add_straight_walk(final_path=final_path, current_path_triangle=current_path_triangle, start_index=4, distance_decrease_multiplier=2)

        return final_borders, final_path

    def add_collision_points_to_lines(self, lines, triangle_point_1, triangle_point_2, triangle_point_3):
        first = True
        while True:
            self.add_triangle_to_list(lines, triangle_point_1, triangle_point_2, triangle_point_3)
            if first:
                self.margin_between_outter_and_inner_triangles = self.line_thickness_with_error_gap / 2
                first = False
            else:
                self.margin_between_outter_and_inner_triangles = self.line_thickness_with_error_gap
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

    def add_triangle_to_list(self, lines, triangle_point_1, triangle_point_2, triangle_point_3):
        self.add_line_to_set(lines, ((triangle_point_1[0], triangle_point_1[1], triangle_point_1[2]),
                                     (triangle_point_2[0], triangle_point_2[1], triangle_point_2[2])))
        self.add_line_to_set(lines, ((triangle_point_2[0], triangle_point_2[1], triangle_point_2[2]),
                                     (triangle_point_3[0], triangle_point_3[1], triangle_point_3[2])))
        self.add_line_to_set(lines, ((triangle_point_3[0], triangle_point_3[1], triangle_point_3[2]),
                                     (triangle_point_1[0], triangle_point_1[1], triangle_point_1[2])))

    def add_line_to_set(self, lines, line_to_add):
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

def movebase_client(path):
    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

    for current_goal in path:
        # Creates a new goal with the MoveBaseGoal constructor
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        # Move to position 0.5 on the x axis of the "map" coordinate frame
        goal.target_pose.pose.position.x = current_goal["position"][0]
        # Move to position 0.5 on the y axis of the "map" coordinate frame
        goal.target_pose.pose.position.y = current_goal["position"][1]
        # No rotation of the mobile base frame w.r.t. map frame
        quat = quaternion_from_euler(0, 0, current_goal["angle"])
        print(quat[0], quat[1], quat[2], quat[3])
        goal.target_pose.pose.orientation.w = quat[3]
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

def initialposeCb(msg):
    initial_pose = msg.pose.pose
    print("initial pose is")
    print("X=" + str(initial_pose.position.x))
    print("y=" + str(initial_pose.position.y))

    # position_x, position_y, position_z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
    # orientation_w, orientation_x, orientation_y, orientation_z = msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z
    # (roll, pitch, yaw) = euler_from_quaternion([orientation_x, orientation_y, orientation_z, orientation_w])
    # # roll, pitch, yaw = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    # print("initialposeCb Position: x: {x}, y: {y}, y: {z}".format(x=position_x, y=position_y, z=position_z))
    # print("initialposeCb Rotation: pitch: {pitch}, roll: {roll}, yaw: {yaw}".format(pitch=pitch, roll=roll, yaw=yaw))
    # print("Orientation: w: {w}, x: {x}, y: {y}, y: {z}".format(w=orientation_w, x=orientation_x, y=orientation_y, z=orientation_z))
    # quat = quaternion_from_euler(roll, pitch, yaw)

def odometryCb(msg):
    position_x, position_y, position_z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
    orientation_w, orientation_x, orientation_y, orientation_z = msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z
    (roll, pitch, yaw) = euler_from_quaternion([orientation_x, orientation_y, orientation_z, orientation_w])

    # roll, pitch, yaw = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    # print("odometryCb Position: x: {x}, y: {y}, y: {z}".format(x=position_x, y=position_y, z=position_z))
    # print("odometryCb Rotation: pitch: {pitch}, roll: {roll}, yaw: {yaw}".format(pitch=pitch, roll=roll, yaw=yaw))
    # print("Orientation: w: {w}, x: {x}, y: {y}, y: {z}".format(w=orientation_w, x=orientation_x, y=orientation_y, z=orientation_z))
    # quat = quaternion_from_euler(roll, pitch, yaw)

# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('get_map_example')
    sub_PoseWithCovarianceStamped = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, initialposeCb)
    # sub_Odometry = rospy.Subscriber('odom', Odometry, odometryCb)

    ms      = MapService()
    occ_map = ms.map_arr

    cb            = CleaningBlocks(occ_map)
    triangle_list = cb.get_triangles()
    # Draw delaunay triangles
    # cb.draw_triangles((0, 255, 0))

    triangles = []
    for t in triangle_list:
        triangles.append((np.array((t[0], t[1], 0)), np.array((t[4], t[5], 0)), np.array((t[2], t[3], 0))))
        # triangles.append((np.array((t[0], t[1], 0)), np.array((t[2], t[3], 0)), np.array((t[4], t[5], 0))))
        # print(triangles[-1])

    # Path planning
    path_finder   = Path_finder()
    borders, path = path_finder.find(triangles)
    print("Done creating the path.")
    print("path length:", len(path))
    # rospy.sleep(1.)
    # rospy.spin()

    # Moves the robot according to the path
    try:
       # Initializes a rospy node to let the SimpleActionClient publish and subscribe
       #  rospy.init_node('movebase_client_py')
        result = movebase_client(path=path)
        if result:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation Exception.")

    # Plots / Saves the path map
    # x = []
    # y = []
    # for i in range(len(path) - 1):
    #     x.append(path[i    ]["position"][0])
    #     y.append(path[i    ]["position"][1])
    #     x.append(path[i + 1]["position"][0])
    #     y.append(path[i + 1]["position"][1])
    #     angle_radian    = path[i]["angle"]
    #     rotation_matrix = R.from_euler('z', angle_radian, degrees=False)
    #     rotated_vector  = rotation_matrix.apply(np.array((0.05, 0.0, 0.0)))
    #     plt.arrow(x=path[i]["position"][0], y=path[i]["position"][1], dx=rotated_vector[0], dy=rotated_vector[1], width=0.5)#.015)
    # for i in range(len(borders)):
    #     plt.plot(np.array(borders[i][0]), np.array(borders[i][1]))
    # plt.plot(np.array(x), np.array(y))
    # plt.annotate(
    #     'Start', xy=(x[0], y[0]), xytext=(x[0], y[0] - 0.5),
    #     horizontalalignment="center",
    #     arrowprops=dict(arrowstyle='->', lw=1)
    # )
    # plt.annotate(
    #     'End', xy=(x[-1], y[-1]), xytext=(x[-1], y[-1] + 0.75),
    #     horizontalalignment="center",
    #     arrowprops=dict(arrowstyle='->', lw=1)
    # )
    # plt.axis('scaled')
    # plt.savefig("path.png")
    # # plt.show()

    # exec_mode = sys.argv[1]
    # print('exec_mode:' + exec_mode)
    #
    # if exec_mode == 'cleaning':
    #     vacuum_cleaning()
    # elif exec_mode == 'inspection':
    #     inspection()
    # else:
    #     print("Code not found")
    #     raise NotImplementedError
