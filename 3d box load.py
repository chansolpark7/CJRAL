import json
from collections import defaultdict, namedtuple
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import random
import visualize
import time
import numpy

from main import *

class Vehicle:
    X = 16
    Y = 28
    Z = 18
    total_volume = X*Y*Z*1000

    def __init__(self, route, OD_matrix):
        self.route = route
        self.cost = 0
        self.used = numpy.zeros((self.X, self.Y, self.Z), dtype=int)
        self.depth = numpy.zeros((self.X, self.Z), dtype=int)
        self.box_list = []
        self.box_num = 0

        self.calculate_cost(OD_matrix)

    def calculate_cost(self, OD_matrix):
        length = len(self.route)
        for i in range(length-1):
            start = self.route[i]
            end = self.route[i+1]
            self. cost += OD_matrix[start][end]

    def load_box_at(self, position, size):
        x, y, z = position
        size_x, size_y, size_z = size
        self.used[x:x+size_x, y:y+size_y, z:z+size_z] = 1
        self.depth[x:x+size_x, z:z+size_z] = numpy.maximum(self.depth[x:x+size_x, z:z+size_z], y+size_y)

    def unload_box_at(self, position, size):
        x, y, z = position
        size_x, size_y, size_z = size
        self.used[x:x+size_x, y:y+size_y, z:z+size_z] = 0
        for dx in range(size_x):
            for dz in range(size_z):
                for depth_y in range(self.Y-1, -1, -1):
                    if self.used[x+dx][depth_y][z+dz]:
                        self.depth[x+dx][z+dz] = depth_y+1
                        break
                else:
                    self.depth[x+dx][z+dz] = 0

    def print_depth(self):
        for z in range(self.Z - 1, -1, -1):
            for x in range(self.X):
                print(f'{self.depth[x][z]:>4d}', end='')
            print()

    def calc_possible_volume(self):
        volume = self.depth.copy()
        for z in range(self.Z-2, -1, -1):
            volume[:, z] = numpy.maximum(self.depth[:, z], volume[:, z+1])
        size_x = 3
        size_y = 3
        size_z = 3
        for x in range(self.X-size_x+1):
            for z in range(self.Z-size_z+1):
                d = volume[x:x+size_x, z:z+size_z].max()
                volume[x:x+size_x, z:z+size_z] = numpy.minimum(volume[x:x+size_x, z:z+size_z], d)
        volume[volume > self.Y - size_y] = self.Y

        return self.total_volume - 1000 * volume.sum()

    def calc_filled_volume(self):
        return self.used.sum()*1000

    def calc_empty_volume(self):
        return self.total_volume - self.calc_filled_volume()

    def load_box_greedy(self, boxes):
        def get_possible_positions(size):
            size_x, size_y, size_z = size
            positions = []
            for x in range(self.X-size_x+1):
                for z in range(self.Z-size_z+1):
                    y = self.depth[x:x+size_x, z:z+size_z].max()
                    if y + size_y > self.Y: continue
                    if z == 0:
                        positions.append((x, y, z))
                    else:
                        if self.used[x:x+size_x, y:y+size_y, z-1].sum() != 0: positions.append((x, y, z))

            return positions

        # 30x40x30, 30x50x40, 50x60x50cm
        # 160*280*180 x y z

        possible_volume_data = []
        empty_volume_data = []

        self.box_num = 0
        for info in boxes:
            best_fit_size = None
            best_fit_position = None
            best_fit_possible_volume = -1
            count = 0
            for size in get_possible_orientations(info):
                positions = get_possible_positions(size)
                count += len(positions)
                for position in positions:
                    self.load_box_at(position, size)
                    possible_volume = self.calc_possible_volume()
                    self.unload_box_at(position, size)
                    if possible_volume > best_fit_possible_volume:
                        best_fit_size = size
                        best_fit_position = position
                        best_fit_possible_volume = possible_volume
            print(f'{count=}')
            if best_fit_position != None:
                self.load_box_at(best_fit_position, best_fit_size)
                self.box_list.append((best_fit_position, best_fit_size))
                self.box_num += 1

                possible_volume_data.append(best_fit_possible_volume)
                empty_volume_data.append(self.calc_empty_volume())
            else: break

        return possible_volume_data, empty_volume_data

def random_boxes(n):
    boxes = []
    for _ in range(n):
        boxes.append(random.randint(0, 2))
    return boxes

def get_possible_orientations(info):
    if info == 0:
        return (3, 3, 4), (3, 4, 3), (4, 3, 3)
    elif info == 1:
        return (3, 4, 5), (3, 5, 4), (4, 3, 5), (4, 5, 3), (5, 3, 4), (5, 4, 3)
    else:
        return (5, 5, 6), (5, 6, 5), (6, 5, 5)

def run():
    destinations, name_to_index, index_to_name = read_map()
    n = len(destinations)
    OD_matrix = read_OD_matrix(n, name_to_index)
    orders = read_orders(n, name_to_index)

    # vehicles = solve_VRP()

    boxes = random_boxes(100)
    # boxes = [0, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2, 1, 1, 2, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 0, 1, 2, 0, 2, 2, 1, 0, 1, 1, 0, 1, 1, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 0, 1, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 0, 2, 0]
    # boxes = [2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0, 2, 2, 2, 0, 1, 0, 0, 2, 2, 2, 0, 0, 2, 0, 1, 0, 0, 2, 2, 2, 1, 0, 2, 2, 1, 1, 2, 2, 2, 0, 2, 1, 1, 1, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 1, 0, 0, 1, 0, 0, 2, 1, 1, 1, 1, 2, 0, 2, 2, 1, 2, 0, 0, 1, 2, 0, 1, 0, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2]
    print(boxes)

    t = time.time()
    v = Vehicle([], OD_matrix)
    possible_volume_data, empty_volume_data = v.load_box_greedy(boxes)
    v.print_depth()
    print('time :', time.time() - t)

    print(v.box_list)
    print(v.box_num)
    
    volume1 = v.calc_empty_volume()
    volume2 = v.calc_filled_volume()
    print(volume1)
    print(volume2)
    print(volume1+volume2, volume1+volume2 == v.total_volume)
    print(v.calc_possible_volume())
    r = v.calc_filled_volume()/v.total_volume
    print(r)

    visualize.graph(possible = possible_volume_data, empty = empty_volume_data)

    viewer = visualize.box_viewer_3d(v.box_list)
    viewer.show()


if __name__ == '__main__':
    run()

# [2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0, 2, 2, 2, 0, 1, 0, 0, 2, 2, 2, 0, 0, 2, 0, 1, 0, 0, 2, 2, 2, 1, 0, 2, 2, 1, 1, 2, 2, 2, 0, 2, 1, 1, 1, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 1, 0, 0, 1, 0, 0, 2, 1, 1, 1, 1, 2, 0, 2, 2, 1, 2, 0, 0, 1, 2, 0, 1, 0, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2]
# [2, 2, 1, 0, 2, 0, 2, 0, 1, 0, 0, 2, 0, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 1, 1, 2, 2, 0, 2, 2, 0, 0, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 2, 1, 1, 2, 0, 1, 2, 1, 0, 0, 1, 2, 2, 0, 0, 1, 0]

# 잘 쌓은 예시
# [0, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2, 1, 1, 2, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 0, 1, 2, 0, 2, 2, 1, 0, 1, 1, 0, 1, 1, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 0, 1, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 0, 2, 0]
