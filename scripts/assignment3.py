#!/usr/bin/env python2.7

import os
import glob
from threading import Timer, Thread, Event

import rospy
import tf
import actionlib
import sys
import time
import math
import numpy as np
# import itertools
import copy
import pickle
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import matplotlib as mpl

from scipy.spatial.transform import Rotation as R
# from scipy.misc import toimage
from scipy import ndimage
from tf.transformations import euler_from_quaternion#, quaternion_from_euler
import dynamic_reconfigure.client
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import multi_move_base
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, PoseStamped, PointStamped
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid, Odometry
from map_msgs.msg import OccupancyGridUpdate
from std_msgs.msg import String
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

def distance(a,b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def is_same_edge(e1, e2):
    return (e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])

def plan_path(first_ind, dist_mat, triangles):
    # print(first_ind)
    # print(triangles)
    # print(dist_mat)
    triangle_order = [triangles[first_ind]]
    visited = [first_ind]
    curr = first_ind
    next_tri = None
    num_of_tri = len(triangles)

    for i in range(num_of_tri - 1):
        min_d = np.inf
        for key, dist in enumerate(dist_mat[curr]):
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

class DecisionMaking():

    def __init__(self, occupancy_map, map_top_left, map_bottom_right, resolution, dirt_piece_radius, save_plot_folder, plot, save_to_file):
        self.results_file_name                                                               = "Results.png"
        self.dirt_pieces_heat_map_file_name                                                  = "1-Dirt_pieces_heat_map.png"
        self.dirt_pieces_heat_and_advisory_to_goal_map_file_name                             = "2-Dirt_pieces_heat_and_advisory_to_goal_map.png"
        self.advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_name        = "3-Advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map.png"
        self.agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_name           = "4-Agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map.png"
        # self.occupancy_agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_name = "6-Occupancy_agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map.png"
        self.agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map_file_name     = "5-Agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map.png"
        self.binary_occupancy_map                                                            = self.occupancy_map_to_binary_map(occupancy_map)
        self.binary_walls_only_occupancy_map                                                 = self.occupancy_map_to_binary_walls_only_map(occupancy_map)
        self.map_top_left                                                                    = map_top_left
        self.map_bottom_right                                                                = map_bottom_right
        self.resolution                                                                      = resolution
        self.dirt_piece_radius                                                               = dirt_piece_radius
        self.dirt_piece_diameter                                                             = self.dirt_piece_radius * 2.0
        self.save_plot_folder                                                                = save_plot_folder
        self.plot                                                                            = plot
        self.save_to_file                                                                    = save_to_file
        self.size_x                                                                          = (self.resolution[1] * abs(self.map_bottom_right[0] - self.map_top_left[0]))
        self.size_y                                                                          = (self.resolution[0] * abs(self.map_bottom_right[1] - self.map_top_left[1]))
        self.map_accessable_min_x                                                            = None
        self.map_accessable_max_x                                                            = None
        self.map_accessable_min_y                                                            = None
        self.map_accessable_max_y                                                            = None
        self.save_plot_path                                                                  = os.path.join(os.getcwd(), self.save_plot_folder)
        self.dirt_pieces_heat_map_file_path                                                  = None
        self.dirt_pieces_heat_and_advisory_to_goal_map_file_path                             = None
        self.agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path           = None
        self.advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path        = None
        # self.occupancy_agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path = None
        self.agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map_file_path     = None
        self.results_file_path                                                               = None
        if self.save_to_file:
            if os.path.isdir(self.save_plot_path):
                shutil.rmtree(self.save_plot_path)
            os.mkdir(self.save_plot_path)
        self.current_save_plot_path  = None
        self.current_iteration_index = 0
        X                            = []
        Y                            = []
        for i in range(self.binary_walls_only_occupancy_map.shape[0]):
            for j in range(self.binary_walls_only_occupancy_map.shape[1]):
                if self.binary_walls_only_occupancy_map[i][j] == 1.0:
                    X.append(j)
                    Y.append(i)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.calculate_min_max_accessable_areas(occupancy_map=occupancy_map)

    def calculate_min_max_accessable_areas(self, occupancy_map):
        self.map_accessable_min_x = self.binary_walls_only_occupancy_map.shape[1]
        self.map_accessable_max_x = -1
        self.map_accessable_min_y = self.binary_walls_only_occupancy_map.shape[0]
        self.map_accessable_max_y = -1
        for i in range(self.binary_walls_only_occupancy_map.shape[0]):
            for j in range(self.binary_walls_only_occupancy_map.shape[1]):
                if occupancy_map[i][j] == 0.0:
                    self.map_accessable_min_x = min(self.map_accessable_min_x, j)
                    self.map_accessable_max_x = max(self.map_accessable_max_x, j)
                    self.map_accessable_min_y = min(self.map_accessable_min_y, i)
                    self.map_accessable_max_y = max(self.map_accessable_max_y, i)

    def occupancy_map_to_binary_map(self, occupancy_map):
        return np.where(occupancy_map < 0.0, 0.0, 1.0)

    def occupancy_map_to_binary_walls_only_map(self, occupancy_map):
        return np.where(occupancy_map <= 0.0, 0.0, 1.0)

    def plot_walls_in_map(self):
        plt.scatter(self.X, self.Y, s=1, c="black")

    def plot_map(self, map_width, map_height, dirt_pieces_locations, our_agent_location, advisory_agent_location, advisory_agent_goal, new_goal_location, title, file_path):
        fig, ax = plt.subplots()
        ax.axes.set_aspect('equal')
        ax.set_ylim(0, map_height)
        ax.set_xlim(0, map_width )
        self.plot_walls_in_map()
        if advisory_agent_goal is not None:
            plt.scatter(np.array([advisory_agent_goal[0]]), np.array(advisory_agent_goal[1]), c="tomato", marker="X", s=200.0, label='Advisory\'s goal')
        if new_goal_location is not None:
            plt.scatter(np.array([new_goal_location[1]]), np.array(new_goal_location[0]), c="dodgerblue", marker="X", s=200.0, label='RESULT agent\'s goal')
        if our_agent_location is not None:
            plt.scatter(np.array([our_agent_location[0]]), np.array(our_agent_location[1]), c="blue", marker=",", s=200.0, label='Agent')
        if advisory_agent_location is not None:
            plt.scatter(np.array([advisory_agent_location[0]]), np.array(advisory_agent_location[1]), c="red", marker=",", s=200.0, label='Advisory')
        if dirt_pieces_locations is not None:
            x_ = []
            y_ = []
            for i in range(len(dirt_pieces_locations)):
                x_.append(dirt_pieces_locations[i][0])
                y_.append(dirt_pieces_locations[i][1])
            x_ = np.array(x_)
            y_ = np.array(y_)
            plt.scatter(x_, y_, c="green", label='Dirt pieces')
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        plt.legend(handles, labels, loc="upper left", framealpha=0.0)
        plt.title(title)
        if self.save_to_file:
            plt.savefig(file_path)
        if self.plot:
            plt.show()
        plt.clf()
        plt.close()

    def save_grid_images_to_file(self):
        dirt_pieces_heat_map                                              = Image.fromarray(np.asarray(Image.open(self.dirt_pieces_heat_map_file_path)))
        dirt_pieces_heat_and_advisory_to_goal_map                         = Image.fromarray(np.asarray(Image.open(self.dirt_pieces_heat_and_advisory_to_goal_map_file_path)))
        agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map       = Image.fromarray(np.asarray(Image.open(self.agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path)))
        advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map    = Image.fromarray(np.asarray(Image.open(self.advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path)))
        agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map = Image.fromarray(np.asarray(Image.open(self.agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map_file_path)))

        dpi      = mpl.rcParams['figure.dpi']
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20,20), dpi=dpi)
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        images = [dirt_pieces_heat_map, dirt_pieces_heat_and_advisory_to_goal_map, advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map, agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map, agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map]
        for i, ax in enumerate(axs.flatten()):
            if i < len(images):
                ax.axis('off')
                ax.imshow(images[i])
            else:
                ax.remove()
        # plt.show()
        plt.savefig(self.results_file_path)
        plt.clf()
        plt.close()

    def find_max_distance_to_accessable_area(self, location):
        max_distance = 0.0
        max_distance = max(max_distance, np.linalg.norm(np.array(((float(self.map_accessable_min_x) / self.resolution[1]), (float(self.map_accessable_min_y) / self.resolution[0]))) - location))
        max_distance = max(max_distance, np.linalg.norm(np.array(((float(self.map_accessable_min_x) / self.resolution[1]), (float(self.map_accessable_max_y) / self.resolution[0]))) - location))
        max_distance = max(max_distance, np.linalg.norm(np.array(((float(self.map_accessable_max_x) / self.resolution[1]), (float(self.map_accessable_min_y) / self.resolution[0]))) - location))
        max_distance = max(max_distance, np.linalg.norm(np.array(((float(self.map_accessable_max_x) / self.resolution[1]), (float(self.map_accessable_max_y) / self.resolution[0]))) - location))
        return max_distance

    def draw_gaussians(self, sigma_x, sigma_y, x, y, z, locations, ascending_weights, aggression, distance_to_goal_factor):
        len_locations = len(locations)
        for i in range(len_locations):
            current_location   = locations[i]
            current_location_x = current_location[0]
            current_location_y = current_location[1]
            z += (distance_to_goal_factor * (aggression * ((1.0 - (ascending_weights * (1.0 - (float(i + 1) / len_locations)))) * (1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-((current_location_x - x) ** 2 / (2 * sigma_x ** 2) + (current_location_y - y) ** 2 / (2 * sigma_y ** 2)))))))

    def create_map(self):
        x      = np.linspace(self.map_top_left[0], self.map_bottom_right[0], self.size_x)
        y      = np.linspace(self.map_top_left[1], self.map_bottom_right[1], self.size_y)
        x, y   = np.meshgrid(x, y)
        z      = np.zeros(shape=x.shape)
        return x, y, z

    def update_heat_map(self, x, y, z, dirt_pieces_locations):
        sigma_x = self.dirt_piece_radius
        sigma_y = self.dirt_piece_radius

        self.draw_gaussians(sigma_x=sigma_x, sigma_y=sigma_y, x=x, y=y, z=z, locations=dirt_pieces_locations, ascending_weights=False, aggression=1.0, distance_to_goal_factor=1.0)

        if self.plot or self.save_to_file:
            z_ = copy.deepcopy(z) * self.binary_occupancy_map
            plt.contourf(x, y, z_, cmap='Blues')
            self.plot_walls_in_map()
            # plt.colorbar()
            plt.title("1-Dirt pieces heat map")
            plt.axis('scaled')
            if self.save_to_file:
                plt.savefig(self.dirt_pieces_heat_map_file_path)
            if self.plot:
                plt.show()
            plt.clf()

        return x, y, z

    def update_advisory_to_goal_map(self, x, y, z, our_agent_location, advisory_agent_location, advisory_agent_goal, number_of_points, aggression):
        if advisory_agent_goal is not None:
            if ((self.dirt_piece_diameter * 6.0) < np.linalg.norm(np.array(our_agent_location) - np.array(advisory_agent_goal))):
                if advisory_agent_location is None:
                    sigma_x   = self.dirt_piece_diameter
                    sigma_y   = self.dirt_piece_diameter
                    locations = []
                    for i in range(2 + 2):
                        locations.append(advisory_agent_goal)
                    distance_to_goal_factor = 1.0

                    self.draw_gaussians(sigma_x=sigma_x, sigma_y=sigma_y, x=x, y=y, z=z, locations=locations, ascending_weights=False, aggression=aggression, distance_to_goal_factor=distance_to_goal_factor)
                else:
                    sigma_x   = self.dirt_piece_radius / 1.5
                    sigma_y   = self.dirt_piece_radius / 1.5
                    x0, y0    = advisory_agent_location[1], advisory_agent_location[0]
                    x1, y1    = advisory_agent_goal[1]    , advisory_agent_goal[0]
                    xs        = np.linspace(x0, x1, number_of_points + 2)
                    ys        = np.linspace(y0, y1, number_of_points + 2)
                    locations = []
                    for i in range(len(xs)):
                        locations.append((ys[i], xs[i]))

                    max_distance            = self.find_max_distance_to_accessable_area(location=advisory_agent_location)
                    distance_to_goal_factor = (np.linalg.norm(np.array((advisory_agent_goal[1], advisory_agent_goal[0])) - np.array((advisory_agent_location[1], advisory_agent_location[0]))) / max_distance)

                    self.draw_gaussians(sigma_x=sigma_x, sigma_y=sigma_y, x=x, y=y, z=z, locations=locations, ascending_weights=True, aggression=aggression, distance_to_goal_factor=distance_to_goal_factor)

        if self.plot or self.save_to_file:
            z_ = copy.deepcopy(z) * self.binary_occupancy_map
            plt.contourf(x, y, z_, cmap='Blues')
            self.plot_walls_in_map()
            # plt.colorbar()
            plt.title("2-Dirt pieces heat and advisory to goal map")
            plt.axis('scaled')
            if self.save_to_file:
                plt.savefig(self.dirt_pieces_heat_and_advisory_to_goal_map_file_path)
            if self.plot:
                plt.show()
            plt.clf()

        return x, y, z

    def update_advisory_distance_to_heat(self, x, y, z, advisory_agent_location):
        if advisory_agent_location is not None:
            advisory_agent_location = np.array(advisory_agent_location)
            z_max                   = np.amax(z)
            max_distance            = self.find_max_distance_to_accessable_area(location=advisory_agent_location)

            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    distance = np.linalg.norm(np.array(((float(j) / self.resolution[1]), (float(i) / self.resolution[0]))) - advisory_agent_location)
                    # if distance < self.dirt_piece_diameter:
                    # # if distance < self.dirt_piece_radius:
                    #     z[i][j] = 0.0
                    # else:
                    # z[i][j] += (z_max * (1.0 - (distance / max_distance)))
                    z[i][j] += (z_max * (distance / max_distance))

        if self.plot or self.save_to_file:
            z_ = copy.deepcopy(z) * self.binary_occupancy_map
            plt.contourf(x, y, z_, cmap='Blues')
            self.plot_walls_in_map()
            # plt.colorbar()
            plt.title("3-Advisory distance to dirt pieces heat and advisory to goal map")
            plt.axis('scaled')
            if self.save_to_file:
                plt.savefig(self.advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path)
            if self.plot:
                plt.show()
            plt.clf()

        return x, y, z

    def update_agent_distance_to_heat(self, x, y, z, our_agent_location):
        our_agent_location = np.array(our_agent_location)
        z_max              = np.amax(z)
        max_distance       = self.find_max_distance_to_accessable_area(location=our_agent_location)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                distance = np.linalg.norm(np.array(((float(j) / self.resolution[1]), (float(i) / self.resolution[0]))) - our_agent_location)
                if distance < self.dirt_piece_diameter:
                # if distance < self.dirt_piece_radius:
                    z[i][j] = 0.0
                else:
                    z[i][j] += (z_max * (1.0 - (distance / max_distance)))

        if self.plot or self.save_to_file:
            z_ = copy.deepcopy(z) * self.binary_occupancy_map
            plt.contourf(x, y, z_, cmap='Blues')
            self.plot_walls_in_map()
            # plt.colorbar()
            plt.title("4-Agent distance to dirt pieces heat and advisory to goal map")
            plt.axis('scaled')
            if self.save_to_file:
                plt.savefig(self.agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path)
            if self.plot:
                plt.show()
            plt.clf()

        return x, y, z

    # def update_occupancy_map(self, x, y, z):
    #     z *= self.binary_occupancy_map
    #
    #     if self.plot or self.save_to_file:
    #         plt.contourf(x, y, z, cmap='Blues')
    #         # plt.colorbar()
    #         plt.title("5-Occupancy/Agent distance to dirt pieces heat and advisory to goal map")
    #         plt.axis('scaled')
    #         if self.save_to_file:
    #             plt.savefig(self.occupancy_agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path)
    #         if self.plot:
    #             plt.show()
    #         plt.clf()
    #
    #     return x, y, z

    def get_indices_of_max(self, z, dirt_pieces_locations, our_agent_location, advisory_agent_location, advisory_agent_goal):
        indices_of_max    = np.unravel_index(z.argmax(), z.shape)
        new_goal_location = ((float(indices_of_max[0]) / self.resolution[1]), (float(indices_of_max[1]) / self.resolution[0]))

        if self.plot or self.save_to_file:
            size_x = abs(self.map_bottom_right[0] - self.map_top_left[0])
            size_y = abs(self.map_bottom_right[1] - self.map_top_left[1])
            self.plot_map(map_width=size_x, map_height=size_y, dirt_pieces_locations=dirt_pieces_locations, our_agent_location=our_agent_location, advisory_agent_location=advisory_agent_location, advisory_agent_goal=advisory_agent_goal, new_goal_location=new_goal_location, title="Agent, RESULT decision, advisory, advisory's goal, and dirt pieces map", file_path=self.agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map_file_path)

        return new_goal_location

    def create_decision_making_map(self, dirt_pieces_locations, our_agent_location, advisory_agent_location, advisory_agent_goal, aggression):
        x, y, z = self.create_map()
        x, y, z = self.update_heat_map(x=x, y=y, z=z, dirt_pieces_locations=dirt_pieces_locations)
        x, y, z = self.update_advisory_to_goal_map(x=x, y=y, z=z, our_agent_location=our_agent_location, advisory_agent_location=advisory_agent_location, advisory_agent_goal=advisory_agent_goal, number_of_points=len(dirt_pieces_locations), aggression=aggression)
        x, y, z = self.update_advisory_distance_to_heat(x=x, y=y, z=z, advisory_agent_location=advisory_agent_location)
        x, y, z = self.update_agent_distance_to_heat(x=x, y=y, z=z, our_agent_location=our_agent_location)
        # x, y, z = self.update_occupancy_map(x=x, y=y, z=z)
        return x, y, z

    def calculate_new_goal(self, dirt_pieces_locations, our_agent_location, advisory_agent_location, advisory_agent_goal, aggression):
        aggression = max(0.0, aggression)
        if self.save_to_file:
            self.current_save_plot_path                                                          = os.path.join(self.save_plot_path, str(self.current_iteration_index))
            self.dirt_pieces_heat_map_file_path                                                  = os.path.join(self.current_save_plot_path, self.dirt_pieces_heat_map_file_name)
            self.dirt_pieces_heat_and_advisory_to_goal_map_file_path                             = os.path.join(self.current_save_plot_path, self.dirt_pieces_heat_and_advisory_to_goal_map_file_name)
            self.agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path           = os.path.join(self.current_save_plot_path, self.agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_name)
            self.advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path        = os.path.join(self.current_save_plot_path, self.advisory_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_name)
            # self.occupancy_agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_path = os.path.join(self.current_save_plot_path, self.occupancy_agent_distance_to_dirt_pieces_heat_and_advisory_to_goal_map_file_name)
            self.agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map_file_path     = os.path.join(self.current_save_plot_path, self.agent_result_decision_advisory_advisorys_goal_and_dirt_pieces_map_file_name)
            self.results_file_path                                                               = os.path.join(self.current_save_plot_path, self.results_file_name)
            os.mkdir(self.current_save_plot_path)
            self.current_iteration_index += 1
        x, y, z        = self.create_decision_making_map(dirt_pieces_locations=dirt_pieces_locations, our_agent_location=our_agent_location, advisory_agent_location=advisory_agent_location, advisory_agent_goal=advisory_agent_goal, aggression=aggression)
        indices_of_max = self.get_indices_of_max(z=z, dirt_pieces_locations=dirt_pieces_locations, our_agent_location=our_agent_location, advisory_agent_location=advisory_agent_location, advisory_agent_goal=advisory_agent_goal)
        if self.save_to_file:
            self.save_grid_images_to_file()
        return (indices_of_max[1], indices_of_max[0])

class CleaningBlocks:

    def __init__(self, ms):
        self.sub_div   = None
        self.corners   = None
        self.triangles = []
        # self.triangle_order = []
        self.occ_map   = ms.map_arr
        self.map_size  = ms.map_arr.shape
        self.rect      = (0, 0, self.map_size[1], self.map_size[0])
        self.graph     = Graph()

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

        ret, thresh  = cv.threshold(self.occ_map, 90, 255, 0)
        thresh       = np.uint8(thresh)
        self.map_rgb = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        corners      = cv.goodFeaturesToTrack(thresh, maxCorners=30, qualityLevel=0.16, minDistance=3, blockSize=6, useHarrisDetector=False)
        # corners = cv.goodFeaturesToTrack(ms.map_binary, 20, 0.01, 10)
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
            # delaunay_color = (255, 0, 0)
            # img = self.map_rgb

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

    # def plan_path(self, first_pose, dist_mat, triangles):
    #
    #     first_ind = self.locate_initial_pose(first_pose)
    #     # dist_mat = self.dist_mat
    #     # dict_vector = []
    #     # for i in range(len(self.triangles)):
    #     #     dist_vector = self.graph.dijkstra(i)
    #     #     dist_mat.append(dist_vector.values())
    #     #     dict_vector.append(dist_vector)
    #     # print(dist_vector)
    #
    #     triangle_order = [triangles[first_ind]]
    #     visited = [first_ind]
    #     curr = first_ind
    #     next_tri = None
    #     num_of_tri = len(triangles)
    #
    #     for i in range(num_of_tri - 1):
    #         min_d = np.inf
    #         for key, dist in dist_mat[curr].items():
    #             if key not in visited and dist < min_d:
    #                 min_d = dist
    #                 next_tri = key
    #         visited.append(next_tri)
    #         triangle_order.append(triangles[next_tri])
    #         curr = next_tri
    #
    #     # while len(dict_vector) is not len(triangle_order):
    #     #     min_d = np.inf
    #     #     next = curr
    #     #     for key, dist in dict_vector[curr].items():
    #     #         if key is not curr and dist < min_d:
    #     #             min_d = dist
    #     #             next = key
    #     #     triangle_order.append(next)
    #     #     for key in dict_vector[curr].keys():
    #     #         if key is not curr:
    #     #             if curr in dict_vector[key]:
    #     #                 dict_vector[key].pop(curr)
    #     #     curr = next
    #     # # print(dist_mat)
    #     # # print(triangle_order)
    #     # self.triangle_order = triangle_order
    #     # sorted_triangles = [None] * len(self.triangle_order)
    #     # j = 0
    #     # for i in self.triangle_order:
    #     #     sorted_triangles[j] = self.triangles[i]
    #     #     j += 1
    #     #
    #     # # self.triangles = sorted_triangles
    #     return triangle_order

# class CleaningBlocks:
#
#     def __init__(self, occ_map):
#         self.triangles = []
#         self.triangle_order = []
#         self.occ_map = occ_map
#         self.map_size = occ_map.shape
#         self.rect = (0, 0, self.map_size[1], self.map_size[0])
#         self.graph = Graph()
#
#         # find corners
#         ret, thresh = cv.threshold(occ_map, 90, 255, 0)
#         thresh = np.uint8(thresh)
#         self.map_rgb = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
#         # map_img_th = thresh.copy()
#         # im2, contours, hierarchy = cv.findContours(map_img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#         # cv.drawContours(map_img_th, contours, -1, (255, 255, 255), 3)
#         # corners = cv.goodFeaturesToTrack(thresh, 25, 0.01, 10)
#         corners = cv.goodFeaturesToTrack(thresh, maxCorners=30, qualityLevel=0.16, minDistance=3, blockSize=6, useHarrisDetector=False)
#         self.corners = np.int0(corners)
#
#         # find triangles
#         # Create an instance of Subdiv2D
#         self.sub_div = cv.Subdiv2D(self.rect)
#         # Insert points into sub_div
#         self.sub_div.insert(corners)
#         # Filter triangles outside the polygon
#         self.extract_triangles()
#
#     def extract_triangles(self):
#         initial_triangles = self.sub_div.getTriangleList()
#
#         r = self.rect
#         filtered_triangles = []
#         for t in initial_triangles:
#             pt1 = (t[0], t[1])
#             pt2 = (t[2], t[3])
#             pt3 = (t[4], t[5])
#             mid1 = (np.uint32((t[1] + t[3]) / 2), np.uint32((t[0] + t[2]) / 2))
#             mid2 = (np.uint32((t[3] + t[5]) / 2), np.uint32((t[2] + t[4]) / 2))
#             mid3 = (np.uint32((t[1] + t[5]) / 2), np.uint32((t[0] + t[4]) / 2))
#             if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
#                 center = (np.round((t[0] + t[2] + t[4]) / 3),np.round((t[1] + t[3] + t[5]) / 3))
#                 center2 = (np.uint32(np.round((t[1] + t[3] + t[5]) / 3)), np.uint32(np.round((t[0] + t[2] + t[4]) / 3)))
#
#                 if self.point_in_room(center2):
#                 # if self.line_in_room(mid1) and self.line_in_room(mid2) and self.line_in_room(mid3):
#                     mat = np.array([[t[0], t[1], 1], [t[2], t[3], 1], [t[4], t[5], 1]])
#                     area = np.linalg.det(mat) / 2
#
#                     tri_edges = [[pt1, pt2, distance(pt1, pt2)], [pt1, pt3, distance(pt1, pt3)],
#                                  [pt2, pt3, distance(pt2, pt3)]]
#                     # center_edges = [[pt1, center, distance(pt1, center)], [pt1, center, distance(pt1, center)],
#                     #                 [pt2, center, distance(pt2, center)]]
#                     self.triangles.append(Triangle(t, center, area, tri_edges))
#                     last_tri_ind = len(self.triangles)-1
#                     self.add_adjacent_tri_edge(last_tri_ind)
#
#     def get_triangles(self):
#         return self.triangles
#
#     # Draw delaunay triangles
#     def draw_triangles(self, delaunay_color):
#         img = self.map_rgb
#         # Draw points
#         for p in self.corners:
#             draw_point(img, p[0], (0, 0, 255))
#
#         for triangle in self.triangles:
#             t = triangle.coordinates
#             pt1 = (t[0], t[1])
#             pt2 = (t[2], t[3])
#             pt3 = (t[4], t[5])
#             cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
#             cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)
#             cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
#             # cv.imshow('delaunay', img)
#             # cv.waitKey(0)
#
#         # Show results
#         cv.imshow('delaunay', img)
#         cv.waitKey(0)
#         # cv.imwrite('delaunay.jpg', img)
#
#     def point_in_room(self, mid_p_i): #Funny - tried to check lines
#         map = self.occ_map
#         if map[mid_p_i] != -1:  # mid pixel
#             return True
#         return False
#
#
#     def add_adjacent_tri_edge(self, last_tri_ind):
#         for i in range(last_tri_ind):
#             if self.is_neighbor(i, last_tri_ind):
#                 a = self.triangles[i].center
#                 b = self.triangles[last_tri_ind].center
#                 self.graph.add_edge(i, last_tri_ind, distance(a, b))
#
#     def is_neighbor(self, v_i, u_i): #Funny
#         v_cor = self.triangles[v_i].coordinates
#         u_cor = self.triangles[u_i].coordinates
#         v_edges = self.triangles[v_i].edges
#         u_edges = self.triangles[u_i].edges
#         # indices = range(0, 6, 2)
#         # for i in indices:
#         #     for j in indices:
#         #         if v_cor[i] == u_cor[j]:
#         #             if v_cor[i+1] == u_cor[j+1]:
#         #                 return True
#         for e1 in v_edges:
#             for e2 in u_edges:
#                 if is_same_edge(e1, e2):
#                     return True
#         return False
#
#     def locate_initial_pose(self, first_pose):
#         min_dist = 1000
#         ind = 0
#         x = first_pose[0]
#         y = first_pose[1]
#         for (i, triangle) in enumerate(self.triangles):
#             c = triangle.center
#             dist = distance(c, first_pose)
#             if dist < min_dist:
#                 min_dist = dist
#                 ind = i
#         return ind
#
#     def draw_triangle_order(self):
#         img = self.map_rgb
#         triangle_order=self.triangle_order
#         for (i, tri) in enumerate(self.triangles):
#             c1 = tri.center
#             c1 = tuple(np.uint32((round(c1[0]), round(c1[1]))))
#
#             if i < len(triangle_order)-1:
#                 c2 = self.triangles[i+1].center
#                 c2 = tuple(np.uint32((round(c2[0]), round(c2[1]))))
#                 cv.line(img, c1, c2, (255, 0, i * 7), 1, cv.LINE_AA, 0)
#
#             cv.circle(img, c1, 2, (255, 0, i * 7), cv.FILLED, cv.LINE_AA)
#
#     def sort(self, first_pose):
#         starting_point_ind = self.locate_initial_pose(first_pose)
#         dist_mat = []
#         dict_vector = []
#         for i in range(len(self.triangles)):
#             dist_vector = self.graph.dijkstra(i)
#             dist_mat.append(dist_vector.values())
#             dict_vector.append(dist_vector)
#             # print(dist_vector)
#
#         triangle_order = [starting_point_ind]
#         curr = starting_point_ind
#         while len(dict_vector) is not len(triangle_order):
#             min_d = np.inf
#             next = curr
#             for key, dist in dict_vector[curr].items():
#                 if key is not curr and dist < min_d:
#                     min_d = dist
#                     next = key
#             triangle_order.append(next)
#             for key in dict_vector[curr].keys():
#                 if key is not curr:
#                     dict_vector[key].pop(curr)
#             curr = next
#         # print(dist_mat)
#         # print(triangle_order)
#         self.triangle_order = triangle_order
#         sorted_triangles = [None] * len(self.triangle_order)
#         j = 0
#         for i in self.triangle_order:
#             sorted_triangles[j] = self.triangles[i]
#             j += 1
#
#         self.triangles = sorted_triangles
#         return self.triangles

class CostMapUpdater:

    def __init__(self):
        self.cost_map = None
        self.shape = None
        rospy.Subscriber('/move_base/global_costmap/costmap'        , OccupancyGrid      , self.init_costmap_callback  )
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

    def __init__(self, our_agent_id):
        self.initial_pose     = None
        self.initial_pose_map = None
        self.advisory_goal    = None
        ag_str                = str(our_agent_id)
        rospy.wait_for_service('tb3_%c/static_map' % ag_str)
        static_map                         = rospy.ServiceProxy('tb3_%c/static_map' % ag_str, GetMap)
        self.sub_PoseWithCovarianceStamped = rospy.Subscriber('tb3_%c/initialpose' % ag_str, PoseWithCovarianceStamped, self.init_pose)
        # rospy.wait_for_service('static_map')
        # static_map = rospy.ServiceProxy('static_map', GetMap)
        # rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.init_pose)

        self.map_data   = static_map().map
        self.map_org    = np.array([self.map_data.info.origin.position.x, self.map_data.info.origin.position.y])
        shape           = self.map_data.info.height, self.map_data.info.width
        self.map_arr    = np.array(self.map_data.data, dtype='float32').reshape(shape)
        self.resolution = self.map_data.info.resolution
        res, map_binary = cv.threshold(self.map_arr, 90, 255, 0)
        self.map_binary = np.uint8(map_binary)
        self.map_rgb    = cv.cvtColor(self.map_binary, cv.COLOR_GRAY2BGR)

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
        self.initial_pose     = msg.pose.pose
        self.initial_pose_map = self.position_to_map(np.array((self.initial_pose.position.x, self.initial_pose.position.y)))
        # print("initial pose is")
        # print("X=" + str(self.initial_pose_map[0]))
        # print("y=" + str(self.initial_pose_map[1]))

    def get_first_pose(self):
        # Waits for YOU to set the initial_pose
        i = 0
        while ms.initial_pose is None:
            if i % 5 == 0:
                print("Waiting for initial_pose. i =", i)
            i += 1
            time.sleep(1.0)
        # print("initial_pose:", ms.initial_pose)

        pos   = self.position_to_map(np.array([self.initial_pose.position.x, self.initial_pose.position.y]))
        angle = euler_from_quaternion((self.initial_pose.orientation.x, self.initial_pose.orientation.y, self.initial_pose.orientation.z, self.initial_pose.orientation.w))[2]
        return pos, angle

    def unregister(self):
        self.sub_PoseWithCovarianceStamped.unregister()


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

# # Based on https://stackabuse.com/dijkstras-algorithm-in-python/
# class Graph:
#     def __init__(self):
#         self.edges = {}
#         self.visited = []
#
#     def add_edges(self, edges):
#         for e in edges:
#             self.add_edge(e[0], e[1], e[2])
#
#     def add_edge(self, u, v, weight):
#         if u in self.edges:
#             self.edges[u].append((v, weight))
#         else:
#             self.edges[u] = [(v, weight)]
#
#         if v in self.edges:
#             self.edges[v].append((u, weight))
#         else:
#             self.edges[v] = [(u, weight)]
#
#     def dijkstra(self, start_vertex):
#         num_vertices = len(self.edges)
#         D = {v: float('inf') for v in range(num_vertices)}
#         D[start_vertex] = 0
#
#         pq = PriorityQueue()
#         pq.put((0, start_vertex))
#
#         while not pq.empty():
#             (dist, current_vertex) = pq.get()
#             self.visited.append(current_vertex)
#
#             for neighbor, dist in self.edges[current_vertex]:
#                 if neighbor not in self.visited:
#                     old_cost = D[neighbor]
#                     new_cost = D[current_vertex] + dist
#                     if new_cost < old_cost:
#                         pq.put((new_cost, neighbor))
#                         D[neighbor] = new_cost
#         self.visited = []
#         return D


class Path_finder:
    def __init__(self, robot_width, error_gap, divide_walk_every):#(10.0 / 3.0)):#robot_width=0.105
        self.error_gap                                 = error_gap
        self.robot_width                               = robot_width
        self.divide_walk_every                         = divide_walk_every
        self.robot_width_with_error_gap                = self.robot_width * (1.0 + max(0.0, self.error_gap))
        self.distance_to_stop_before_next_triangle     = self.robot_width_with_error_gap
        self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap

    def find_inspection(self, triangle_list):
        final_borders = []
        final_path    = []

        triangles = []
        for triangle in triangle_list:
            t = triangle.coordinates
            triangles.append((np.array((t[0], t[1], 0)), np.array((t[4], t[5], 0)), np.array((t[2], t[3], 0))))

        path = []
        for triangle in triangles:
            current_inner_triangles = []
            self.add_collision_points_to_lines(current_inner_triangles, triangle[0], triangle[1], triangle[2], only_one_iteration=True)
            self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap
            while 3 < len(current_inner_triangles):
                del current_inner_triangles[-1]
            triangle_center = ((triangle[0] + triangle[1] + triangle[2]) / 3.0)
            current_inner_triangles.append(triangle_center        )
            path                   .append(current_inner_triangles)

        current_path_triangles = []
        for i in range(len(path)):
            current_path = path[i]
            j            = 0
            x            = []
            y            = []
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
            if j == 0:
                final_borders.append((x, y))
            x = copy.deepcopy(x)
            y = copy.deepcopy(y)
            x.append(current_path[j + 3][0])
            y.append(current_path[j + 3][1])
            current_path_triangles.append((x, y))

        for j in range(len(current_path_triangles)):
            current_path_triangle = current_path_triangles[j]
            if j == 0:
                yaw_angle = 0.0
            else:
                yaw_angle = math.atan2(current_path_triangles[j][1][6] - current_path_triangles[j - 1][1][6], current_path_triangles[j][0][6] - current_path_triangles[j - 1][0][6])
            final_path.append({"position": (current_path_triangle[0][6], current_path_triangle[1][6], 0.0), "angle": yaw_angle})

        return final_borders, final_path

    def find_cleaning(self, triangles):
        triangles = self.sort_vertices_by_prev_center_to_closest_next_vertex(triangles)

        path = []
        for triangle in triangles:
            current_inner_triangles = []
            self.add_collision_points_to_lines(current_inner_triangles, triangle[0], triangle[1], triangle[2], only_one_iteration=False)
            self.margin_between_outter_and_inner_triangles = self.robot_width_with_error_gap
            path.append(current_inner_triangles)

        final_borders        = []
        final_path           = []
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
                final_path.append({"position": (new_direction_vector[0], new_direction_vector[1], 0.0), "angle": yaw_angle})
                i = 1.0
                while i < number_of_segments:
                    new_direction_vector = np.array((current_path_triangle[0][start_index], current_path_triangle[1][start_index], 0)) + direction_vector_norm * min((i * self.divide_walk_every), direction_vector_new_len)
                    final_path.append({"position": (new_direction_vector[0], new_direction_vector[1], 0.0), "angle": yaw_angle})
                    i += 1.0
                new_direction_vector = np.array((current_path_triangle[0][start_index], current_path_triangle[1][start_index], 0)) + direction_vector_norm * direction_vector_new_len
                final_path.append({"position": (new_direction_vector[0], new_direction_vector[1], 0.0), "angle": yaw_angle})

            if len(current_path_triangles) == 1:
                current_path_triangle = current_path_triangles[0]
                center_x              = (current_path_triangle[0][0] + current_path_triangle[0][2] + current_path_triangle[0][4]) / 3.0
                center_y              = (current_path_triangle[1][0] + current_path_triangle[1][2] + current_path_triangle[1][4]) / 3.0
                final_path.append({"position": (center_x, center_y, 0.0), "angle": final_path[-1]["angle"]})
            else:
                for j in range(1, len(current_path_triangles)):
                    current_path_triangle = current_path_triangles[j]
                    yaw_angle             = math.atan2(current_path_triangle[1][1] - current_path_triangle[1][0], current_path_triangle[0][1] - current_path_triangle[0][0])
                    final_path.append({"position": (current_path_triangle[0][0], current_path_triangle[1][0], 0.0), "angle": yaw_angle})
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

    def ptInTriang(self, p_test, p0, p1, p2):
        dX = p_test[0] - p0[0]
        dY = p_test[1] - p0[1]
        dX20 = p2[0] - p0[0]
        dY20 = p2[1] - p0[1]
        dX10 = p1[0] - p0[0]
        dY10 = p1[1] - p0[1]
        s_p = (dY20 * dX) - (dX20 * dY)
        t_p = (dX10 * dY) - (dY10 * dX)
        D = (dX10 * dY20) - (dY10 * dX20)
        if D > 0:
            return ((s_p >= 0) and (t_p >= 0) and (s_p + t_p) <= D)
        else:
            return ((s_p <= 0) and (t_p <= 0) and (s_p + t_p) >= D)

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
            if inner_triangle_point_1 is None or not self.ptInTriang(p_test=inner_triangle_point_1, p0=triangle_point_1, p1=triangle_point_3, p2=triangle_point_2):
                return
            inner_triangle_point_2 = self.find_inner_triangle_point(triangle_point_2, triangle_point_1, triangle_point_3)
            if inner_triangle_point_2 is None or not self.ptInTriang(p_test=inner_triangle_point_2, p0=triangle_point_2, p1=triangle_point_1, p2=triangle_point_3):
                return
            inner_triangle_point_3 = self.find_inner_triangle_point(triangle_point_3, triangle_point_2, triangle_point_1)
            if inner_triangle_point_3 is None or not self.ptInTriang(p_test=inner_triangle_point_3, p0=triangle_point_3, p1=triangle_point_2, p2=triangle_point_1):
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

def movebase_client_cleaning(map_service, path):
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

class DummyCleaner:

    def __init__(self, ms, id):

        first_pose, first_angle = ms.get_first_pose()
        pose = ms.map_to_position(first_pose)
        self.pose_x = pose[0]
        self.pose_y = pose[1]
        self.odomSub = rospy.Subscriber('/tb3_%d/odom' % int(id), Odometry, self.update_dirt_status)
        self.dirtPieces = DirtPiecesService(ms=ms)

        rc_DWA_client = dynamic_reconfigure.client.Client('tb3_%d/move_base/DWAPlannerROS' % int(id))
        rc_DWA_client.update_configuration({"max_vel_x": "0.22"})
        rc_DWA_client.update_configuration({"yaw_goal_tolerance": "inf"})
        rc_DWA_client.update_configuration({"xy_goal_tolerance": "0.25"})

    def update_dirt_status(self, msg):
        # parsed_id = (msg.header.frame_id.replace('tb3_', '')).replace('/odom', '')
        # agent_id = int(parsed_id)
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y

    def run(self):
        while not rospy.is_shutdown():
            if self.dirtPieces.coordinates_list is None:
                time.sleep(1)
                print("waiting for dirt pieces")
                continue
            if not self.dirtPieces.coordinates_list:
                self.dirtPieces.unregister()
                self.odomSub.unregister()
                break
            closest_dirt = None
            min_dist = np.Inf
            for cur_dirt in self.dirtPieces.coordinates_list:
                d = ms.map_to_position(cur_dirt)
                # euclidean distance
                dist = math.pow((d[0] - self.pose_x), 2) + math.pow((d[1] - self.pose_y), 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_dirt = d

            try:
                result = multi_move_base.move(agent_id=int(agent_id), x=closest_dirt[0], y=closest_dirt[1])
                if result:
                    rospy.loginfo("Goal execution done!")
            except rospy.ROSInterruptException:
                rospy.loginfo("Navigation Exception.")

def move_robot_on_path_cleaning(map_service, path):
    try:
       # Initializes a rospy node to let the SimpleActionClient publish and subscribe
        result = movebase_client_cleaning(map_service=map_service, path=path)
        if result:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation Exception.")


class AdvisoryGoalService(object):

    def __init__(self, ms, adv_agent_id):
        self.ms            = ms
        self.adv_agent_id  = int(adv_agent_id)
        self.advisory_goal = None

        # Advisory's current goal subscriber
        self.sub_PoseStamped = rospy.Subscriber('tb3_%d/move_base/current_goal' % int(self.adv_agent_id), PoseStamped, self.advisory_goal_cb)

    def advisory_goal_cb(self, msg):
        self.advisory_goal = self.ms.position_to_map(np.array((msg.pose.position.x, msg.pose.position.y)))

    def unregister(self):
        self.sub_PoseStamped.unregister()


class DirtPiecesService(object):

    def __init__(self, ms):
        self.ms                         = ms
        self.coordinates_list           = None
        self.coordinates_list_timestamp = None

        # Dirt pieces subscriber
        self.sub_String = rospy.Subscriber('dirt', String, self.dirtCb)

    def dirtCb(self, msg):
        # print("dirtCb()")
        # print(msg)
        # First batch is send as tuples, so if there are any '()' replace with '[]'
        new_string               = msg.data.replace(')', ']').replace('(', '[')
        msg_list                 = new_string.split("][")
        # msg_list                 = msg.data.split("][")
        msg_list[0]              = msg_list[0][1:]
        msg_list[-1]             = msg_list[-1][:-1]
        current_coordinates_list = []
        for i in range(len(msg_list)):
            current_coordinates = msg_list[i].split(", ")
            if 0 < len(current_coordinates[0]):
                current_coordinates_list.append(self.ms.position_to_map((float(current_coordinates[0]), float(current_coordinates[1]))))
        self.coordinates_list           = current_coordinates_list
        self.coordinates_list_timestamp = time.time()
        # print("dirtCb() end")

    def unregister(self):
        self.sub_String.unregister()

class DecisionMakingTimer():
    def __init__(self, t, decisionMaking, dirtPiecesService, advisoryGoalService, our_agent_location, advisory_agent_location, agentPositionService, aggression):
        self.finished                         = False
        self.t                                = t
        self.decisionMaking                   = decisionMaking
        self.dirtPiecesService                = dirtPiecesService
        self.advisoryGoalService              = advisoryGoalService
        self.our_agent_location               = our_agent_location
        self.advisory_agent_location          = advisory_agent_location
        self.agentPositionService             = agentPositionService
        self.aggression                       = aggression
        self.client                           = actionlib.SimpleActionClient('tb3_%d/move_base' % 0, MoveBaseAction)
        self.client.wait_for_server()
        self.thread                           = Timer(self.t, self.handle_function, [])
        self.prev_dirt_pieces_locations_len   = None
        self.prev_our_agent_location          = None
        self.prev_advisory_agent_location     = None
        self.prev_current_advisory_agent_goal = None
        self.first_iteration                  = True

    def handle_function(self, *args):
        should_continue = self.run()
        if should_continue:
            self.thread = Timer(self.t, self.handle_function, [])
            self.thread.start()

    def run(self):
        if self.dirtPiecesService.coordinates_list is not None:
            if not self.dirtPiecesService.coordinates_list:
                self.cancel()
                return False
            if self.first_iteration:
                self.prev_dirt_pieces_locations_len   = len(self.dirtPiecesService.coordinates_list)
                self.prev_our_agent_location          = self.our_agent_location
                self.prev_advisory_agent_location     = self.advisory_agent_location  # (165.0, 165.0)
                self.prev_current_advisory_agent_goal = self.advisoryGoalService.advisory_goal # (200.0, 125.0)
            else:
                self.agentPositionService.receive_agent_pose()
                self.our_agent_location      = self.agentPositionService.our_agent_position
                self.advisory_agent_location = self.agentPositionService.adv_agent_position
            if ((len(self.dirtPiecesService.coordinates_list) != self.prev_dirt_pieces_locations_len     ) or
                    (tuple(self.our_agent_location)           != tuple(self.prev_our_agent_location     )) or
                    (((self.advisory_agent_location is not None) and (self.prev_advisory_agent_location is not None)) and (tuple(self.advisory_agent_location)           != tuple(self.prev_advisory_agent_location))) or
                    (((self.advisoryGoalService.advisory_goal is not None) and (self.prev_current_advisory_agent_goal is not None)) and (tuple(self.advisoryGoalService.advisory_goal) != tuple(self.prev_current_advisory_agent_goal))) or
                    self.first_iteration                                                                   ):
                self.first_iteration                  = False
                self.prev_dirt_pieces_locations_len   = len(self.dirtPiecesService.coordinates_list)
                self.prev_our_agent_location          = self.our_agent_location
                self.prev_advisory_agent_location     = self.advisory_agent_location
                self.prev_current_advisory_agent_goal = self.advisoryGoalService.advisory_goal # (200.0, 125.0)
                new_goal_map                          = self.decisionMaking.calculate_new_goal(dirt_pieces_locations=self.dirtPiecesService.coordinates_list, our_agent_location=self.prev_our_agent_location, advisory_agent_location=self.prev_advisory_agent_location, advisory_agent_goal=self.prev_current_advisory_agent_goal, aggression=self.aggression)
                new_goal_position                     = ms.map_to_position(np.array((new_goal_map[0], new_goal_map[1])))
                try:
                    self.client.cancel_goal()
                    result = self.our_move(agent_id=int(agent_id), x=new_goal_position[0], y=new_goal_position[1])
                    # result = multi_move_base.move(agent_id=int(agent_id), x=new_goal_position[0], y=new_goal_position[1])
                    if result:
                        rospy.loginfo("Goal execution done!")
                except rospy.ROSInterruptException:
                    rospy.loginfo("Navigation Exception.")
                # self.our_agent_location = new_goal_map
        return True

    def our_movebase_client(self, agent_id=0, x=1.0, y=0.5, w=1.0):
        # print("Sending goal")
        # Creates a new goal with the MoveBaseGoal constructor
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        # Move 0.5 meters forward along the x axis of the "map" coordinate frame
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        # No rotation of the mobile base frame w.r.t. map frame
        goal.target_pose.pose.orientation.w = w
        # Sends the goal to the action server.
        self.client.send_goal(goal)
        # print("Sent goal")

    def our_move(self, agent_id=0, x=1.0, y=0.5, w=1.0):
        try:
            result = self.our_movebase_client(agent_id, x, y, w)
            if result:
                rospy.loginfo("Goal execution done!")
            else:
                rospy.loginfo("Goal unachievable!")
            return result
        except rospy.ROSInterruptException:
            rospy.loginfo("Navigation test finished.")

    def start(self):
        self.thread.start()

    def cancel(self):
        self.finished = True
        self.thread.cancel()
        self.dirtPiecesService  .unregister()
        self.advisoryGoalService.unregister()
        # rospy.signal_shutdown("BYE")

# noinspection PyTypeChecker,PyUnresolvedReferences
def vacuum_cleaning(ms, our_agent_id, adv_agent_id, robot_width, error_gap, dirt_piece_radius):
    print('Starting vacuum_cleaning()')

    rc_DWA_client = dynamic_reconfigure.client.Client('tb3_%d/move_base/DWAPlannerROS' % int(our_agent_id))
    rc_DWA_client.update_configuration({"max_vel_x": "0.22"})
    rc_DWA_client.update_configuration({"yaw_goal_tolerance": "inf"})
    rc_DWA_client.update_configuration({"xy_goal_tolerance": "0.25"})

    # plot                    = True
    plot                    = False
    save_to_file            = True
    # save_to_file            = False
    occupancy_map           = ms.map_arr
    map_top_left            = (0, 0)
    map_bottom_right        = (occupancy_map.shape[1], occupancy_map.shape[0])
    resolution              = (1, 1) # (10, 10)
    aggression              = 1.0  ### 0.0 = Ignore advisory. 1.0 = Normal aggression. The higher = The more our agent tries to block the advisory.
    decisionMaking          = DecisionMaking(occupancy_map=occupancy_map, map_top_left=map_top_left, map_bottom_right=map_bottom_right, resolution=resolution, dirt_piece_radius=dirt_piece_radius, save_plot_folder="plots", plot=plot, save_to_file=save_to_file)
    dirtPiecesService       = DirtPiecesService(ms=ms)
    advisoryGoalService     = AdvisoryGoalService(ms=ms, adv_agent_id=adv_agent_id)
    agentPositionService    = AgentPositionService(ms=ms, our_agent_id=our_agent_id, adv_agent_id=adv_agent_id)
    first_pose, first_angle = ms.get_first_pose()
    our_agent_location      = first_pose # (125.0, 125.0)
    advisory_agent_location = agentPositionService.adv_agent_position
    r                       = rospy.Rate(10)

    t = DecisionMakingTimer(0.0001, decisionMaking, dirtPiecesService, advisoryGoalService, our_agent_location, advisory_agent_location, agentPositionService, aggression)
    t.start()

    while not t.finished:
        r.sleep()


### INSPECTION ###

path_folder_name                     = "paths"
suspicious_points_map_folder_name    = "suspicious_points_maps"
found_circles_maps_folder_name       = "found_circles_maps"
differences_maps_folder_name         = "differences_maps"
differences_maps_process_folder_name = "differences_maps_process"
current_path_index                   = 0
current_maps_index                   = 0
differences_map_index                = 0

class AgentPositionService:
    def __init__(self, ms, our_agent_id, adv_agent_id):
        self.ms                 = ms
        self.our_agent_id       = int(our_agent_id)
        self.adv_agent_id       = int(adv_agent_id)
        self.our_agent_position = None
        self.our_agent_rotation = None
        self.adv_agent_position = None
        self.adv_agent_rotation = None
        self.listener           = tf.TransformListener()

    def receive_agent_pose(self):
        try:
            (trans_our, rot_our)    = self.listener.lookupTransform('/map', 'tb3_%d/base_link' % self.our_agent_id, rospy.Time(0))
            (trans_adv, rot_adv)    = self.listener.lookupTransform('/map', 'tb3_%d/base_link' % self.adv_agent_id, rospy.Time(0))
            self.our_agent_position = self.ms.position_to_map(np.array((trans_our[0], trans_our[1])))
            self.our_agent_rotation = rot_our
            self.adv_agent_position = self.ms.position_to_map(np.array((trans_adv[0], trans_adv[1])))
            self.adv_agent_rotation = rot_adv
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("EXCEPTION: receive_agent_pose")

class AgentPositionTimer:
    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function, [])

    def handle_function(self, *args):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function, [])
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()

class InspectionCostmapUpdater:
    def __init__(self, occ_map, agent_id, spheres_diameter, spheres_filter_size, sparsity):
        self.spheres_diameter                       = spheres_diameter
        self.spheres_filter_size                    = spheres_filter_size # Odd number
        self.sparsity                               = sparsity
        self.occ_map_original                       = occ_map
        # self.occ_map_binary_dilation                = self.binary_dilation(map=self.occ_map_original, iterations1=0, iterations2=3)
        self.occ_map_binary_dilation                = self.binary_dilation(map=self.occ_map_original, iterations1=0, iterations2=4)
        self.cost_map                               = None
        self.cost_map_binary                        = None
        self.differences_map                        = None
        self.differences_map_occupied_pixels_counts = []
        self.shape                                  = None
        self.current_index_in_path                  = 1#0
        self.updated_index_in_path                  = -1
        self.filter_diagonal                        = (2.0 * (self.spheres_filter_size ** 2.0)) ** 0.5
        self.agent_id_int                           = int(agent_id)
        rospy.Subscriber('tb3_%d/move_base/global_costmap/costmap'        %self.agent_id_int, OccupancyGrid      , self.init_costmap_callback  )
        rospy.Subscriber('tb3_%d/move_base/global_costmap/costmap_updates'%self.agent_id_int, OccupancyGridUpdate, self.costmap_callback_update)

    def init_costmap_callback(self, msg):
        self.shape    = msg.info.height, msg.info.width
        self.cost_map = np.array(msg.data).reshape(self.shape)

    def costmap_callback_update(self, msg):
        if self.updated_index_in_path < self.current_index_in_path:
            shape = msg.height, msg.width
            data  = np.array(msg.data).reshape(shape)
            self.cost_map[msg.y:msg.y + shape[0], msg.x: msg.x + shape[1]] = data
            self.updated_index_in_path = self.current_index_in_path

    def update_index_in_path(self, index):
        self.current_index_in_path = index

    def calculate_differences_map(self):
        cost_map_ = np.where(self.cost_map < 85, 0, self.cost_map)  # self.cost_map < 90
        self.cost_map_binary = self.map_to_binary_map(map=cost_map_)
        self.differences_map = self.cost_map_binary - self.occ_map_binary_dilation
        self.differences_map = np.where(self.differences_map < 0.0, 0.0, self.differences_map)
        self.differences_map = self.binary_dilation(map=self.differences_map, iterations1=3, iterations2=2)

    def get_suspicious_points(self, other_agent_position, robot_width, error_gap, plot=False, save_plot_to_file=False):
        global suspicious_points_map_folder_name
        global current_maps_index
        result = []
        self.calculate_differences_map()
        if self.differences_map is not None:
            differences_map       = copy.deepcopy(self.differences_map)
            suspicious_points_map = np.zeros(shape=differences_map.shape)
            height                = differences_map.shape[0]
            width                 = differences_map.shape[1]
            half_filter           = int(self.spheres_filter_size / 2.0)
            for i in range(0, height, self.sparsity):
                for j in range(0, width, self.sparsity):
                    i_  = i - half_filter
                    j_  = j - half_filter
                    sum = 0.0
                    for k in range(self.spheres_filter_size):
                        for l in range(self.spheres_filter_size):
                            if ((0 <= (i_ + k) < height) and (0 <= (j_ + l) < width)):
                                sum += differences_map[i_ + k][j_ + l]
                    suspicious_points_map[i][j] = sum

            max_filter_map = None
            if plot or save_plot_to_file:
                max_filter_map = np.zeros(shape=suspicious_points_map.shape)

            suspicious_coorinations = []
            height_filters          = int(suspicious_points_map.shape[0] / self.spheres_filter_size)
            width_filters           = int(suspicious_points_map.shape[1] / self.spheres_filter_size)
            for i in range(height_filters):
                for j in range(width_filters):
                    i_                           = i * self.spheres_filter_size
                    j_                           = j * self.spheres_filter_size
                    current_max                  = -1.0
                    current_max_x, current_max_y = -1, -1
                    for k in range(self.spheres_filter_size):
                        for l in range(self.spheres_filter_size):
                            current_value = suspicious_points_map[i_ + k][j_ + l]
                            if ((0.0 < current_value) and (current_max < current_value)):
                                current_max                  = current_value
                                current_max_x, current_max_y = j_ + l, i_ + k
                    if 0.0 < current_max:
                        suspicious_coorinations.append(((current_max_y, current_max_x, 0.0), current_max))
                        if plot or save_plot_to_file:
                            for k in range(3):
                                for l in range(3):
                                    max_filter_map[current_max_y - 1 + k][current_max_x - 1 + l] = current_max

            suspicious_coorinations_to_keep = []
            while ((suspicious_coorinations is not None) and (0 < len(suspicious_coorinations))):
                indices_to_remove                       = []
                max_suspicious_coorination              = max(suspicious_coorinations, key=lambda x: x[1])
                index_of_max_suspicious_coorination     = suspicious_coorinations.index(max_suspicious_coorination)
                max_suspicious_coorination_coordination = np.array(max_suspicious_coorination[0])
                suspicious_coorinations_to_keep.append(((max_suspicious_coorination[0][1], max_suspicious_coorination[0][0], 0.0), max_suspicious_coorination[1]))
                indices_to_remove              .append(index_of_max_suspicious_coorination                                                                       )
                for i in range(len(suspicious_coorinations)):
                    if index_of_max_suspicious_coorination != i:
                        suspicious_coorination_to_compare = suspicious_coorinations[i]
                        distance                          = np.linalg.norm(max_suspicious_coorination_coordination - np.array(suspicious_coorination_to_compare[0]))
                        if distance <= self.filter_diagonal:
                            indices_to_remove.append(i)
                for i in sorted(indices_to_remove, reverse=True):
                    del suspicious_coorinations[i]

            # Removes suspicious points which are too close to the other agent
            suspicious_coorinations_without_other_agent = []
            other_agent_position                        = np.array(other_agent_position)
            radius_from_other_agent_position            = ((float(robot_width) * (1.0 + float(error_gap))) / 2.0)
            for i in range(len(suspicious_coorinations_to_keep)):
                current_suspicious_coorination       = suspicious_coorinations_to_keep[i]
                current_suspicious_coorination_array = np.array((current_suspicious_coorination[0][1], current_suspicious_coorination[0][0]))
                if radius_from_other_agent_position < np.linalg.norm(other_agent_position - current_suspicious_coorination_array):
                    suspicious_coorinations_without_other_agent.append(current_suspicious_coorination)

            result = suspicious_coorinations_without_other_agent

            if plot or save_plot_to_file:
                max_filter_map_filtered = np.zeros(shape=suspicious_points_map.shape)
                for i in range(len(suspicious_coorinations_without_other_agent)):
                    suspicious_coorination = suspicious_coorinations_without_other_agent[i]
                    for k in range(3):
                        for l in range(3):
                            max_filter_map_filtered[suspicious_coorination[0][1] - 1 + k][suspicious_coorination[0][0] - 1 + l] = suspicious_coorination[1]
                fig, axs = plt.subplots(1, 3)
                images = [suspicious_points_map, max_filter_map, max_filter_map_filtered]
                images_titles = ["filter_map", "max_filter_map", "max_filter_map_filtered"]
                for i, ax in enumerate(axs.flatten()):
                    if i < len(images):
                        ax.set_title(images_titles[i])
                        ax.imshow(images[i])
                    else:
                        ax.remove()
                # plt.show()
                current_folder = os.path.join(suspicious_points_map_folder_name, str(self.agent_id_int))
                if not os.path.isdir(current_folder):
                    os.mkdir(current_folder)
                # create_folder(folder_name=current_folder)
                plt.savefig(os.path.join(current_folder, str(current_maps_index) + ".png"))
                plt.clf()
                current_maps_index += 1

        return result

    def detect_spheres(self, agentPositionService):
        global differences_maps_folder_name
        global differences_map_index
        current_differences_map_file = os.path.join(differences_maps_folder_name, str(differences_map_index) + ".png")
        result                       = None
        self.calculate_differences_map()
        if self.differences_map is not None:
            differences_map = copy.deepcopy(self.differences_map)
            plt.clf()
            plt.imshow(differences_map)
            plt.savefig(current_differences_map_file)
            plt.clf()

            # Loads an image
            src = cv.imread(cv.samples.findFile(current_differences_map_file), cv.IMREAD_COLOR)
            if src is None: # Check if image is loaded fine
                print ('Error opening image!')
                print ('Usage: hough_circle.py [image_name -- default ' + current_differences_map_file + '] \n')
                return -1
            gray    = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            gray    = cv.medianBlur(gray, 5)
            # plt.imshow(gray)
            # plt.show()
            rows    = gray.shape[0]
            circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,#/ 8
                                      param1=100, param2=9,# param1=100, param2=30,
                                      minRadius=9, maxRadius=21)# minRadius=1, maxRadius=30)
            if circles is not None:
                agent_0_position                 = np.array(agentPositionService.agent_0_position)
                agent_1_position                 = np.array(agentPositionService.agent_1_position)
                radius_from_other_agent_position = ((self.spheres_diameter + float(robot_width) * (1.0 + float(error_gap))) / 2.0)
                circles_not_agents_indices       = []
                index                            = 0
                for i in circles[0, :]:
                    center              = (i[0], i[1])
                    circle_center_array = np.array((center[1], center[0]))
                    if ((radius_from_other_agent_position < np.linalg.norm(agent_0_position - circle_center_array)) and
                            (radius_from_other_agent_position < np.linalg.norm(agent_1_position - circle_center_array))):
                        circles_not_agents_indices.append(index)
                    index += 1
                new_circles = np.empty(shape=(circles.shape[0], len(circles_not_agents_indices), circles.shape[2]))
                for i in range(len(circles_not_agents_indices)):
                    new_circles[0][i] = circles[0][circles_not_agents_indices[i]]
                circles = new_circles
                circles = np.uint16(np.around(circles))
                list_of_circles_centers = []
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    radius = i[2]  # circle outline
                    list_of_circles_centers.append((center, radius))
                # if save_plot_to_file:
                #     for i in circles[0, :]:
                #         center = (i[0], i[1])
                #         cv.circle(src, center, 1, (0, 100, 100), 3) # circle center
                #         radius = i[2] # circle outline
                #         cv.circle(src, center, radius, (255, 0, 255), 3)
                #     # cv.imshow("detected circles" + str(len(circles)), src)
                #     # cv.waitKey(0)
                #     cv.imwrite(current_found_circles_map_file, src)
                # result = len(circles[0])
                result = list_of_circles_centers

            # if save_plot_to_file:
            #     # if 0 < result:
            #     if result is not None:
            #         found_circles_map = cv.imread(current_found_circles_map_file)
            #     else:
            #         found_circles_map = self.differences_map
            #     fig, axs          = plt.subplots(2, 3)
            #     images            = [self.occ_map_original, self.cost_map, self.differences_map,
            #                          self.occ_map_binary_dilation, self.cost_map_binary, found_circles_map]
            #     images_titles     = ["occ", "cost", "differences",
            #                          "occ_binary_dilation", "cost_filtered_binary", "found_circles"]
            #     for i, ax in enumerate(axs.flatten()):
            #         if i < len(images):
            #             ax.set_title(images_titles[i])
            #             ax.imshow(images[i])
            #         else:
            #             ax.remove()
            #     # plt.show()
            #     plt.savefig(current_differences_map_process_file)
            #     plt.clf()

        return result

    def binary_dilation(self, map, iterations1, iterations2):
        occ_map_ = self.map_to_binary_map(map=map)
        if 0 < iterations1:
            struct1 = ndimage.generate_binary_structure(2, 2)
            occ_map_ = ndimage.binary_dilation(occ_map_, structure=struct1, iterations=iterations1).astype(occ_map_.dtype)
        if 0 < iterations2:
            struct2  = ndimage.generate_binary_structure(2, 1)
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

    def show_map(self):
        if not self.cost_map is None:
            plt.imshow(self.differences_map)
            # plt.imshow(self.cost_map)
            plt.show()
            plt.clf()

# def create_surrounding_sphere(map_service, surround_times, radius):
#     surrounding_sphere = np.empty(shape=(surround_times, 3))
#     one_time_angle     = (np.pi * 2.0) / surround_times
#     # base_vector        = map_service.map_to_position(np.array((radius, 0.0)))
#     # base_vector        = np.array((base_vector[0], base_vector[1], 0.0))
#     base_vector        = np.array((radius, 0.0, 0.0))
#     for i in range(surround_times):
#         angle_radian            = one_time_angle * i
#         rotation_matrix         = R.from_euler('z', angle_radian, degrees=False)
#         rotated_vector          = rotation_matrix.apply(base_vector)
#         # rotated_vector_position = map_service.map_to_position(np.array((rotated_vector[0], rotated_vector[1])))
#         # surrounding_sphere[i]   = np.array((rotated_vector_position[0], rotated_vector_position[1], 0.0))
#         surrounding_sphere[i]   = rotated_vector
#     return surrounding_sphere

def save_image_map_with_path(map_service, path):
    global current_path_index
    map_with_path = copy.deepcopy(map_service.map_arr)
    plt.imshow(map_with_path)
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
        plt.arrow(x=path[i]["position"][0], y=path[i]["position"][1], dx=rotated_vector[0], dy=rotated_vector[1], width=0.5)  # .015)
    plt.plot(np.array(x), np.array(y))
    plt.axis('scaled')
    plt.title('Path. Index:' + str(current_path_index))
    plt.savefig(os.path.join(path_folder_name, str(current_path_index) + ".png"))
    plt.clf()
    current_path_index += 1

class Agent1SpheresTimer():
    def __init__(self, t, hFunction, pub, icmu, agentPositionService):
        self.start_time        = rospy.Time.now() #time.time()
        self.pub               = pub
        self.t                 = t
        self.hFunction         = hFunction
        self.thread            = Timer(self.t, self.handle_function, [icmu, agentPositionService])

    def handle_function(self, *args):
        icmu                          = args[0]
        agentPositionService          = args[1]
        self.run_and_publish(icmu, agentPositionService)
        self.thread = Timer(self.t, self.handle_function, [icmu, agentPositionService])
        self.thread.start()

    def run_and_publish(self, icmu, agentPositionService):
        global differences_map_index
        list_of_detected_spheres       = self.hFunction(icmu, agentPositionService)
        differences_map_index         += 1
        pickle_list_of_circles_centers = pickle.dumps(list_of_detected_spheres)
        self.pub.publish(pickle_list_of_circles_centers)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()

list_of_detected_spheres_agent_1 = None

class Agent0SpheresTimer():
    def __init__(self, t, hFunction, minimum_distance_between_spheres_centers, pub, icmu, agentPositionService, save_number_of_circles_in_map):
        self.start_time                               = rospy.Time.now() #time.time()
        self.pub                                      = pub
        self.t                                        = t
        self.hFunction                                = hFunction
        self.minimum_distance_between_spheres_centers = minimum_distance_between_spheres_centers
        self.thread                                   = Timer(self.t, self.handle_function, [icmu, agentPositionService, save_number_of_circles_in_map])
        self.number_of_circles                        = None

    def handle_function(self, *args):
        icmu                          = args[0]
        agentPositionService          = args[1]
        save_number_of_circles_in_map = args[2]
        self.run_and_publish(icmu, agentPositionService, save_number_of_circles_in_map)
        self.thread = Timer(self.t, self.handle_function, [icmu, agentPositionService, save_number_of_circles_in_map])
        self.thread.start()

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def run_and_publish(self, icmu, agentPositionService, save_number_of_spheres_in_map):
        global list_of_detected_spheres_agent_1
        global found_circles_maps_folder_name
        global differences_maps_process_folder_name
        global differences_maps_folder_name
        global differences_map_index
        list_of_detected_spheres_        = []
        list_of_detected_spheres_agent_0 = self.hFunction(icmu, agentPositionService)
        if list_of_detected_spheres_agent_0 is not None:
            for i in range(len(list_of_detected_spheres_agent_0)):
                list_of_detected_spheres_.append(list_of_detected_spheres_agent_0[i])
        if list_of_detected_spheres_agent_1 is not None:
            duplicate_spheres_centers_indices = []
            if list_of_detected_spheres_agent_0 is not None:
                for i in range(len(list_of_detected_spheres_agent_1)):
                    current_sphere_center_agent_1 = np.array(list_of_detected_spheres_agent_1[i][0])
                    for j in range(len(list_of_detected_spheres_agent_0)):
                        current_sphere_center_agent_0 = np.array(list_of_detected_spheres_agent_0[j][0])
                        if np.linalg.norm(current_sphere_center_agent_0 - current_sphere_center_agent_1) < self.minimum_distance_between_spheres_centers:
                            duplicate_spheres_centers_indices.append(i)
            for i in range(len(list_of_detected_spheres_agent_1)):
                if i not in duplicate_spheres_centers_indices:
                    list_of_detected_spheres_.append(list_of_detected_spheres_agent_1[i])

        # list_of_detected_spheres = list_of_detected_spheres_
        list_of_detected_spheres          = []
        duplicate_spheres_centers_indices = []
        for i in range(len(list_of_detected_spheres)):
            current_sphere_center_1 = np.array(list_of_detected_spheres_[i][0])
            for j in range((i + 1), len(list_of_detected_spheres_)):
                current_sphere_center_0 = np.array(list_of_detected_spheres_[j][0])
                if np.linalg.norm(current_sphere_center_0 - current_sphere_center_1) < self.minimum_distance_between_spheres_centers:
                    duplicate_spheres_centers_indices.append(i)
        for i in range(len(list_of_detected_spheres_)):
            if i not in duplicate_spheres_centers_indices:
                list_of_detected_spheres.append(list_of_detected_spheres_[i])

        if save_number_of_spheres_in_map:
            current_found_circles_map_file       = os.path.join(found_circles_maps_folder_name      , str(differences_map_index) + ".png")
            current_differences_map_process_file = os.path.join(differences_maps_process_folder_name, str(differences_map_index) + ".png")
            current_differences_map_file         = os.path.join(differences_maps_folder_name        , str(differences_map_index) + ".png")
            differences_map_index               += 1
            src                                  = cv.imread(cv.samples.findFile(current_differences_map_file), cv.IMREAD_COLOR)
            for i in range(len(list_of_detected_spheres)):
                center = (list_of_detected_spheres[i][0][0], list_of_detected_spheres[i][0][1])
                cv.circle(src, center, 1, (0, 100, 100), 3)  # circle center
                radius = list_of_detected_spheres[i][1]  # circle outline
                cv.circle(src, center, radius, (255, 0, 255), 3)
                # cv.imshow("detected circles" + str(len(circles)), src)
                # cv.waitKey(0)
                cv.imwrite(current_found_circles_map_file, src)
            if list_of_detected_spheres is None:
                found_circles_map = icmu.differences_map
            else:
                found_circles_map = cv.imread(current_found_circles_map_file)
            fig, axs = plt.subplots(2, 3)
            images = [icmu.occ_map_original, icmu.cost_map, icmu.differences_map,
                      icmu.occ_map_binary_dilation, icmu.cost_map_binary, found_circles_map]
            images_titles = ["occ", "cost", "differences",
                             "occ_binary_dilation", "cost_filtered_binary", "found_circles"]
            for i, ax in enumerate(axs.flatten()):
                if i < len(images):
                    ax.set_title(images_titles[i])
                    ax.imshow(images[i])
                else:
                    ax.remove()
            # plt.show()
            plt.savefig(current_differences_map_process_file)
            plt.clf()

        self.number_of_spheres = len(list_of_detected_spheres)
        ts                     = (rospy.Time.now() - self.start_time)
        report_string_msg      = "{X} spheres detected at time {TIMESTAMP}".format(X=self.number_of_spheres, TIMESTAMP=ts)
        self.pub.publish(report_string_msg)
        print(report_string_msg)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()

def detect_spheres(icmu, agentPositionService):
    list_of_detected_spheres = icmu.detect_spheres(agentPositionService=agentPositionService)
    return list_of_detected_spheres

def movebase_client_inspection(map_service, agent_id, path):
    result = None
    for current_goal in path:
        position = map_service.map_to_position(indices=np.array((current_goal["position"][0], current_goal["position"][1])))
        try:
            result = multi_move_base.move(agent_id=int(agent_id), x=position[0], y=position[1])
            if result:
                rospy.loginfo("Goal execution done!")
        except rospy.ROSInterruptException:
            rospy.loginfo("Navigation Exception.")
    return result

agent_1_finished     = False
sub_agent_1_finished = False

def agent_1_finished_cb(msg):
    global agent_1_finished
    print("agent_1_finished_cb")
    agent_1_finished = True

def inspection_report_agent_1_cb(msg):
    global list_of_detected_spheres_agent_1
    list_of_detected_spheres_agent_1 = pickle.loads(msg.data)

suspicious_points = []

def move_robot_on_path_inspection(map_service, agent_id, path, robot_width, error_gap, spheres_diameter, save_map_with_path_image, save_number_of_circles_in_map):
    global agent_1_finished
    global suspicious_points
    global sub_agent_1_finished
    if save_map_with_path_image:
        save_image_map_with_path(map_service=map_service, path=path)
    spheres_filter_size = 25#41
    sparsity            = 5
    # surround_times      = 6
    filter_diagonal     = (2.0 * (spheres_filter_size ** 2.0)) ** 0.5
    # surrounding_sphere  = create_surrounding_sphere(map_service=map_service, surround_times=surround_times, radius=spheres_filter_size)
    # surrounding_sphere  = create_surrounding_sphere(map_service=map_service, surround_times=surround_times, radius=((filter_diagonal / 2.0) + (2.0 * sparsity) + (robot_width * (1.0 + error_gap))))
    icmu                = InspectionCostmapUpdater(occ_map=map_service.map_arr, agent_id=agent_id, spheres_diameter=spheres_diameter, spheres_filter_size=spheres_filter_size, sparsity=sparsity)

    agentPositionService = AgentPositionService(ms=ms)
    agentPositionTimer   = AgentPositionTimer(1, agentPositionService.receive_agent_pose)
    agentPositionTimer.start()

    pub_inspection_report         = None
    pub_inspection_report_agent_1 = None
    sub_inspection_report_agent_1 = None
    t                             = None
    if agent_id == '0':
        sub_inspection_report_agent_1            = rospy.Subscriber('inspection_report_agent_1', String, inspection_report_agent_1_cb)
        pub_inspection_report                    = rospy.Publisher('inspection_report', String, queue_size=1)
        minimum_distance_between_spheres_centers = spheres_diameter
        t                                        = Agent0SpheresTimer(30, detect_spheres, minimum_distance_between_spheres_centers, pub_inspection_report, icmu, agentPositionService, save_number_of_circles_in_map)
        t.start()
    else:
        pub_inspection_report_agent_1 = rospy.Publisher('inspection_report_agent_1', String, queue_size=1)
        t                             = Agent1SpheresTimer(5, detect_spheres, pub_inspection_report_agent_1, icmu, agentPositionService)
        t.start()

    def take_path_step(map_service, agent_id, path):
        try:
            result = movebase_client_inspection(map_service=map_service, agent_id=agent_id, path=path)
            if result:
                rospy.loginfo("Goal execution done!")
        except rospy.ROSInterruptException:
            rospy.loginfo("Navigation Exception.")

    i = 1
    while i < len(path):
        icmu.update_index_in_path(index=i)
        while ((icmu.updated_index_in_path < i) or ((agentPositionService.agent_0_position is None) or (agentPositionService.agent_1_position is None))):
            time.sleep(0.01)

        if agent_id == '0':
            other_agent_position = agentPositionService.agent_1_position
        else:
            other_agent_position = agentPositionService.agent_0_position
        # current_suspicious_points = icmu.get_suspicious_points(other_agent_position=other_agent_position, robot_width=robot_width, error_gap=error_gap, plot=False, save_plot_to_file=True)
        current_suspicious_points = icmu.get_suspicious_points(other_agent_position=other_agent_position, robot_width=robot_width, error_gap=error_gap, plot=False, save_plot_to_file=False)

        # Removes suspicious points that have been checked previously
        indices_to_remove = set()
        for j in range(len(current_suspicious_points)):
            suspicious_point = np.array(current_suspicious_points[j][0])
            for k in range(len(suspicious_points)):
                previous_suspicious_point = np.array(suspicious_points[k])
                # if np.linalg.norm(suspicious_point - previous_suspicious_point) <= filter_diagonal:
                if np.linalg.norm(suspicious_point - previous_suspicious_point) <= spheres_filter_size:
                    indices_to_remove.add(j)
        indices_to_remove = sorted(list(indices_to_remove))
        map(lambda x: current_suspicious_points.pop(x), sorted(indices_to_remove, key=lambda x: -x))
        for j in range(len(current_suspicious_points)):
            suspicious_points.append(current_suspicious_points[j][0])

        # Removes path goals if they're too close to any suspicious sphere point
        # if 0 < len(list_of_suspicious_points):
        path_points_indices_to_remove = set()
        for j in range(len(current_suspicious_points)):
            suspicious_point = np.array(current_suspicious_points[j][0])
            for k in range(i, len(path)):
                path_point = path[k]["position"]
                if np.linalg.norm(np.array((path_point[0], path_point[1], 0.0)) - suspicious_point) < ((filter_diagonal / 2.0) + sparsity):
                    path_points_indices_to_remove.add(k)
        path_points_indices_to_remove = list(reversed(sorted(path_points_indices_to_remove)))
        for j in range(len(path_points_indices_to_remove)):
            del path[path_points_indices_to_remove[j]]

        if save_map_with_path_image:
            save_image_map_with_path(map_service=map_service, path=path)

        # Takes the next step of the path
        if i < len(path):
            current_path = [path[i]]
            take_path_step(map_service=map_service, agent_id=agent_id, path=current_path)

        i += 1

    # suspicious_points_path = []
    # suspicious_points      = list(reversed(suspicious_points))
    #
    # # Filters out points which are not in the room
    # def is_point_in_room(occ_map, current_surrounding_point, radius):
    #     radius = np.ceil(radius)
    #     height = occ_map.shape[0]
    #     width = occ_map.shape[1]
    #     diameter = int(math.ceil(radius * 2))
    #     base_i = int(current_surrounding_point[1] - radius)
    #     base_j = int(current_surrounding_point[0] - radius)
    #     for i in range(diameter):
    #         for j in range(diameter):
    #             current_i = (base_i + i)
    #             current_j = (base_j + j)
    #             if ((0 <= current_i < height) and (0 <= current_j < width)):
    #                 if np.linalg.norm(np.array((current_j, current_i, 0.0)) - current_surrounding_point) <= radius:
    #                     if occ_map[current_i][current_j] == -1.0:
    #                         return False
    #     return True
    #
    # # Creates a list such that each suspicious point has a list of surrounding points
    # suspicious_points_surrounding_points = []
    # for j in range(len(suspicious_points)):
    #     suspicious_point           = np.array(suspicious_points[j][0])
    #     current_surrounding_sphere = surrounding_sphere + suspicious_point
    #     current_surrounding_points = []
    #     for k in range(surround_times):
    #         current_surrounding_point = current_surrounding_sphere[k]
    #         if is_point_in_room(occ_map=icmu.occ_map_original, current_surrounding_point=current_surrounding_point, radius=(robot_width * error_gap)):
    #             # if is_point_in_room(cost_map_binary=icmu.cost_map_binary, current_surrounding_point=current_surrounding_point, radius=(robot_width * error_gap)):
    #             #     print("In")
    #             current_surrounding_points.append(current_surrounding_point)
    #         # else:
    #         #     print("Out")
    #         #     current_surrounding_points.append(None)
    #     suspicious_points_surrounding_points.append(current_surrounding_points)
    #
    # # Creates a list of paths which go around the suspicious points, starting from closest points
    # surrounding_points_paths                        = []
    # current_robot_position                          = np.array(path[i - 1]["position"])
    # # current_robot_position                          = np.array(path[-1]["position"])
    # suspicious_points_closest_surrounding_point_len = len(suspicious_points_surrounding_points)
    # for j in range(suspicious_points_closest_surrounding_point_len):
    #     suspicious_points_closest_surrounding_point = []
    #     for k in range(len(suspicious_points_surrounding_points)):
    #         current_surrounding_points = suspicious_points_surrounding_points[k]
    #         if 0 < len(current_surrounding_points):
    #             min_distance_index = 0
    #             min_distance       = np.linalg.norm(current_surrounding_points[min_distance_index] - current_robot_position)
    #             for l in range(1, len(current_surrounding_points)):
    #                 current_surrounding_point = current_surrounding_points[l]
    #                 current_distance          = np.linalg.norm(current_surrounding_point - current_robot_position)
    #                 if current_distance < min_distance:
    #                     min_distance_index = l
    #                     min_distance       = current_distance
    #             suspicious_points_closest_surrounding_point.append((k, min_distance_index, min_distance))
    #     if 0 < len(suspicious_points_closest_surrounding_point):
    #         suspicious_points_closest_surrounding_point              = sorted(suspicious_points_closest_surrounding_point, key=lambda x: x[2])  # key=lambda x: x[1]
    #         closest_suspicious_point_index                           = suspicious_points_closest_surrounding_point[0]
    #         closest_suspicious_point_surrounding_points_index        = closest_suspicious_point_index[0]  # k
    #         closest_suspicious_point_closest_surrounding_point_index = closest_suspicious_point_index[1]  # min_distance_index
    #         closest_suspicious_point_surrounding_points              = suspicious_points_surrounding_points[closest_suspicious_point_surrounding_points_index]
    #         closest_suspicious_point_closest_surrounding_point       = closest_suspicious_point_surrounding_points[closest_suspicious_point_closest_surrounding_point_index]
    #         closest_suspicious_point_surrounding_points_             = copy.deepcopy(closest_suspicious_point_surrounding_points)
    #         del closest_suspicious_point_surrounding_points_[closest_suspicious_point_closest_surrounding_point_index]
    #         current_surrounding_points_path                 = [closest_suspicious_point_closest_surrounding_point]
    #         closest_suspicious_point_surrounding_points_len = len(closest_suspicious_point_surrounding_points_)
    #         for k in range(closest_suspicious_point_surrounding_points_len):
    #             min_distance_index = 0
    #             min_distance       = np.linalg.norm(closest_suspicious_point_surrounding_points_[0] - current_surrounding_points_path[-1])
    #             for l in range(1, len(closest_suspicious_point_surrounding_points_)):
    #                 closest_suspicious_point_surrounding_point = closest_suspicious_point_surrounding_points_[l]
    #                 current_distance                           = np.linalg.norm(closest_suspicious_point_surrounding_point - current_surrounding_points_path[-1])
    #                 if current_distance < min_distance:
    #                     min_distance_index = l
    #                     min_distance       = current_distance
    #             current_surrounding_points_path.append(closest_suspicious_point_surrounding_points_[min_distance_index])
    #             del closest_suspicious_point_surrounding_points_[min_distance_index]
    #         surrounding_points_paths.append(current_surrounding_points_path)
    #         current_robot_position = current_surrounding_points_path[-1]
    #         del suspicious_points_surrounding_points[closest_suspicious_point_surrounding_points_index]
    #
    # for j in range(len(surrounding_points_paths)):
    #     surrounding_points_path = surrounding_points_paths[j]
    #     surrounding_points_path = list(reversed(surrounding_points_path))
    #     for k in range(len(surrounding_points_path)):
    #         path.insert(i, {"position": (surrounding_points_path[k][0], surrounding_points_path[k][1], surrounding_points_path[k][2]), "angle": 0.0})
    #
    # path.extend(suspicious_points_path)
    #
    # while i < len(path):
    #     icmu.update_index_in_path(index=i)
    #     while icmu.updated_index_in_path < i:
    #         time.sleep(0.01)
    #
    #     number_of_circles = icmu.calculate_number_of_circles_in_map(save_plot_to_file=save_number_of_circles_in_map)
    #     print("number_of_circles:", number_of_circles)
    #
    #     if save_map_with_path_image:
    #         save_image_map_with_path(map_service=map_service, path=path)
    #
    #     # Takes the next step of the path
    #     current_path = [path[i]]
    #     take_path_step(map_service=map_service, path=current_path)
    #
    #     i += 1

    # Unregisters publishers
    agentPositionTimer.cancel()
    if agent_id == '0':
        rate = rospy.Rate(1)
        while not agent_1_finished:
            rate.sleep()
        # noinspection PyUnresolvedReferences
        # sub_agent_1_finished.unregister()
        # sub_agent_1_finished = None
        t.cancel()
        t.run_and_publish(icmu=icmu, agentPositionService=agentPositionService, save_number_of_spheres_in_map=save_number_of_circles_in_map)
        # pub_inspection_report.unregister()
        # pub_inspection_report = None
        # sub_inspection_report_agent_1.unregister()
        # sub_inspection_report_agent_1 = None
    else:
        t.cancel()
        t.run_and_publish(icmu=icmu, agentPositionService=agentPositionService)
        # pub_inspection_report_agent_1.unregister()
        # pub_inspection_report_agent_1 = None

def create_folder(folder_name):
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    # try:
    #     os.makedirs(folder_name)
    # except OSError:
    #     if not os.path.isdir(folder_name):
    #         raise
    # # Deletes all files in folder
    # files = glob.glob(folder_name + '/*')
    # for f in files:
    #     os.remove(f)

def create_paths_images_folder():
    global path_folder_name
    global suspicious_points_map_folder_name
    global found_circles_maps_folder_name
    global differences_maps_folder_name
    global differences_maps_process_folder_name
    create_folder(folder_name=path_folder_name                    )
    create_folder(folder_name=suspicious_points_map_folder_name   )
    create_folder(folder_name=found_circles_maps_folder_name      )
    create_folder(folder_name=differences_maps_folder_name        )
    create_folder(folder_name=differences_maps_process_folder_name)

def init_master_agent():
    cb              = CleaningBlocks(ms)
    dist_mat_np     = np.array(cb.dist_mat)
    extreme_nodes   = np.squeeze(np.where(np.max(dist_mat_np) == dist_mat_np))
    nodes_indices_0 = []
    nodes_indices_1 = []
    for i in range(len(cb.triangles)):
        dist_0 = dist_mat_np[i][extreme_nodes[0]]
        dist_1 = dist_mat_np[i][extreme_nodes[1]]
        if dist_0 < dist_1:
            nodes_indices_0.append(i)
        else:
            nodes_indices_1.append(i)

    dist_mat_0      = dist_mat_np[nodes_indices_0, :][:, nodes_indices_0]
    dist_mat_1      = dist_mat_np[nodes_indices_1, :][:, nodes_indices_1]
    triangles_0     = [cb.triangles[i] for i in nodes_indices_0]
    triangles_1     = [cb.triangles[i] for i in nodes_indices_1]
    first_ind_0     = nodes_indices_0.index(extreme_nodes[0])
    first_ind_1     = nodes_indices_1.index(extreme_nodes[1])
    triangle_list_0 = plan_path(first_ind=first_ind_0, dist_mat=dist_mat_0, triangles=triangles_0)
    triangle_list_1 = plan_path(first_ind=first_ind_1, dist_mat=dist_mat_1, triangles=triangles_1)
    # cb.draw_path(nodes_indices_1)
    # cb.draw_triangles((0, 255, 0), cb.triangles)

    return triangle_list_0, triangle_list_1

pub_triangle_list     = None

def publish_triangle_list(pub_triangle_list, pickle_triangle_list):
    pub_triangle_list.publish(pickle_triangle_list)

class triangle_listPerpetualTimer():
    def __init__(self, t, hFunction, pub, pickle_triangle_list):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function, [pub, pickle_triangle_list])

    def handle_function(self, *args):
        pub                  = args[0]
        pickle_triangle_list = args[1]
        self.hFunction(pub, pickle_triangle_list)
        self.thread = Timer(self.t, self.handle_function, [pub, pickle_triangle_list])
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()

def broadcast_triangle_list(triangle_list):
    global pub_triangle_list
    print("broadcast_triangle_list")
    pickle_triangle_list = pickle.dumps(triangle_list)
    pub_triangle_list    = rospy.Publisher('triangle_list', String, queue_size=1)
    t                    = triangle_listPerpetualTimer(1, publish_triangle_list, pub_triangle_list, pickle_triangle_list)
    t.start()
    return t

sub_triangle_list     = None
triangle_list_agent_1 = None

# noinspection PyUnresolvedReferences
def triangle_list_cb(msg):
    global triangle_list_agent_1
    print("triangle_list_cb")
    sub_triangle_list.unregister()
    triangle_list_agent_1 = pickle.loads(msg.data)

def receive_triangle_list():
    global sub_triangle_list
    global triangle_list_agent_1
    print("receive_triangle_list")
    sub_triangle_list = rospy.Subscriber('triangle_list', String, triangle_list_cb)
    i = 0
    while triangle_list_agent_1 is None:
        if i % 50 == 0:
            print("Waiting to receive the triangle_list from agent_0")
        i += 1
        time.sleep(0.1)

def inspection(ms, agent_id, agent_max_vel, robot_width, error_gap, spheres_diameter):
    global triangle_list_agent_1
    global sub_agent_1_finished
    print('start inspection')

    create_paths_images_folder()

    rc_DWA_client = dynamic_reconfigure.client.Client('tb3_%d/move_base/DWAPlannerROS' % int(agent_id))
    rc_DWA_client.update_configuration({"max_vel_x": agent_max_vel})

    triangle_list = None
    timer         = None
    if agent_id == '0':
        sub_agent_1_finished             = rospy.Subscriber("agent_1_finished", String, agent_1_finished_cb)
        triangle_list_0, triangle_list_1 = init_master_agent()
        timer = broadcast_triangle_list(triangle_list=triangle_list_1)
        triangle_list = triangle_list_0
    elif agent_id == '1':
        receive_triangle_list()
        triangle_list = triangle_list_agent_1
    else:
        print("ERROR: Bad agent's ID.")
        exit(-1)

    first_pose, first_angle = ms.get_first_pose()

    # Path planning
    path_finder   = Path_finder(robot_width=robot_width, error_gap=error_gap, divide_walk_every=20.0)
    borders, path = path_finder.find_inspection(triangle_list=triangle_list)
    path.insert(0, {"position": (first_pose[0], first_pose[1], 0.0), "angle": first_angle})
    print("Done creating the path. Length:", len(path))

    # Plots / Saves the path map
    plot_path(borders=borders, path=path, plot=False, save_to_file=False)
    # plot_path(borders=borders, path=path, plot=False, save_to_file=True)

    # Moves the robot according to the path
    move_robot_on_path_inspection(map_service=ms, agent_id=agent_id, path=path, robot_width=robot_width, error_gap=error_gap, spheres_diameter=spheres_diameter, save_map_with_path_image=True, save_number_of_circles_in_map=True)
    # move_robot_on_path_inspection(map_service=ms, agent_id=agent_id, path=path, robot_width=robot_width, error_gap=error_gap, spheres_diameter=spheres_diameter, save_map_with_path_image=False, save_number_of_circles_in_map=False)

    if agent_id == '0':
        timer.cancel()
    elif agent_id == '1':
        pub_agent_1_finished = rospy.Publisher('agent_1_finished', String, queue_size=1)
        rate                 = rospy.Rate(0.2)
        for i in range(10):
            pub_agent_1_finished.publish("Finished")
            rate.sleep()
        pub_agent_1_finished.unregister()


if __name__ == '__main__':
    robot_width       = 5.0
    error_gap         = 0.15
    dirt_piece_radius = 0.5
    spheres_diameter  = 20.0
    # spheres_diameter  = 5.0

    # Initializes a rospy node to let the SimpleActionClient publish and subscribe
    # rospy.init_node('assignment3') # rospy.init_node('get_map_example')
    # rc_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
    # rc_DWA_client.update_configuration({"max_vel_x": np.inf})
    # rc_DWA_client.update_configuration({"max_vel_trans": np.inf})
    # configurations = rc_DWA_client.get_configuration()

    exec_mode = sys.argv[1]
    print('exec_mode:' + exec_mode)

    agent_id = sys.argv[2]
    print('agent id:' + agent_id)

    our_agent_id = agent_id
    adv_agent_id = str(1 - int(agent_id))

    if agent_id == '0':
        rospy.init_node('assignment3_agent_0')
    elif agent_id == '1':
        rospy.init_node('assignment3_agent_1')
    else:
        print("ERROR: Bad agent's ID.")
        exit(-1)

    Triangle = namedtuple('Triangle', ['coordinates', 'center', 'area', 'edges'])

    ms = MapService(our_agent_id=our_agent_id)

    if exec_mode == 'cleaning':
        # TODO: to be deleted
        if agent_id == '1':
            dc = DummyCleaner(ms=ms, id=adv_agent_id)
            dc.run()
        else:
            vacuum_cleaning(ms=ms, our_agent_id=our_agent_id, adv_agent_id=adv_agent_id, robot_width=robot_width, error_gap=error_gap, dirt_piece_radius=dirt_piece_radius)
        # vacuum_cleaning(ms=ms, our_agent_id=our_agent_id, adv_agent_id=adv_agent_id, robot_width=robot_width, error_gap=error_gap, dirt_piece_radius=dirt_piece_radius)
    elif exec_mode == 'inspection':
        agent_max_vel = sys.argv[3]
        print('agent max vel:' + agent_max_vel)
        inspection(ms=ms, agent_id=agent_id, agent_max_vel=agent_max_vel, robot_width=robot_width, error_gap=error_gap, spheres_diameter=spheres_diameter)
    else:
        print("ERROR: exec_mode not found")
        raise NotImplementedError
    ms.unregister()

    ##### Gridsearch for the corners detection algorithm's parameters

    # ms_boolean = np.zeros(ms.map_arr.shape)
    # result = np.zeros(ms.map_arr.shape)
    #
    # for i in range(ms.map_arr.shape[0]):
    #     for j in range(ms.map_arr.shape[1]):
    #         if ms.map_arr[i][j] == 100.0:
    #             ms_boolean[i][j] = 1.0
    #
    # # inspectionCostmapUpdater = InspectionCostmapUpdater()
    # # ms_boolean = inspectionCostmapUpdater.binary_dilation(map=ms_boolean, iterations1=0, iterations2=5)
    #
    # im = np.array(ms_boolean * 255, dtype=np.uint8)
    # ms_boolean = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
    #
    # good_parameters = []
    #
    # maxCornerss   = [30]
    # qualityLevels = [(0.01 * (x + 1)) for x in range(10, 20)]# range(0, 40)]
    # minDistances  = [x for x in range(2, 25)]
    # blockSizes    = [x for x in range(2, 10)]
    #
    # all_combintations = list(itertools.product(*[maxCornerss, qualityLevels, minDistances, blockSizes]))
    #
    # k = 0.0
    # for i, combintation in enumerate(all_combintations):
    #     maxCorners   = combintation[0]
    #     qualityLevel = combintation[1]
    #     minDistance  = combintation[2]
    #     blockSize    = combintation[3]
    #
    #     corners = cv.goodFeaturesToTrack(ms_boolean, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize, useHarrisDetector=False)
    #
    #     corners = np.int0(corners)
    #     if len(corners) == 14:
    #         good_parameters.append((maxCorners, qualityLevel, minDistance, blockSize))
    #     k += 1
    #     print("Done", (k / len(all_combintations)))
    #
    # print("len(good_parameters)", len(good_parameters))
    #
    # with open('good_parameters.pkl', 'wb') as outp:
    #     pickle.dump(good_parameters, outp, pickle.HIGHEST_PROTOCOL)
    #
    # with open('good_parameters.pkl', 'rb') as inp:
    #     good_parameters = pickle.load(inp)
    #
    # for i in range(len(good_parameters)):
    #     current_good_parameters = good_parameters[i]
    #     result = np.zeros(ms.map_arr.shape)
    #
    #     corners = cv.goodFeaturesToTrack(ms_boolean, maxCorners=current_good_parameters[0], qualityLevel=current_good_parameters[1], minDistance=current_good_parameters[2], blockSize=current_good_parameters[3], useHarrisDetector=False)
    #     corners = np.int0(corners)
    #     for i in corners:
    #         x, y = i.ravel()
    #         cv.circle(result, (x, y), 3, 255, -1)
    #
    #     plt.imshow(result)
    #     plt.title('Parameters: ' + str(current_good_parameters[0]) + ", " + str(current_good_parameters[1]) + ", " + str(current_good_parameters[2]) + ", " + str(current_good_parameters[3]))
    #     plt.show()
    #
    # # ms.show_map()