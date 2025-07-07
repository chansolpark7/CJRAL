import json
from collections import defaultdict, namedtuple
# from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import random
import visualize
import time
import numpy

box_info = [
    [30, 40, 30],
    [30, 50, 40],
    [50, 60, 50]
]

Point = namedtuple('Point', ['longitude', 'latitude'])

def read_map():
    with open('Data_Set.json', 'rt', encoding='utf-8') as file:
        raw_data = json.load(file)

    depot = Point(**raw_data['depot']['location'])
    destinations = dict()
    for destination in raw_data['destinations']:
        destinations[destination['destination_id']] = Point(**destination['location'])

    return depot, destinations

def read_OD_matrix():
    with open('distance-data.txt', 'rt', encoding='utf-8') as file:
        file.readline()
        OD_matrix = defaultdict(dict)
        while True:
            string = file.readline().rstrip()
            if not string: break
            origin, destination, time_min, distance_meter = string.split()
            time_min = float(time_min)
            distance_meter = int(distance_meter)
            OD_matrix[origin][destination] = (time_min, distance_meter)

    return OD_matrix

class Vehicle:
    X = 16
    Y = 28
    Z = 18
    total_volume = X*Y*Z*1000

    def __init__(self, route, OD_matrix):
        self.route = route
        self.cost = 0
        self.used = [[[False]*self.Z for _ in range(self.Y)] for _ in range(self.X)]
        self.depth = [[0] * self.Z for _ in range(self.X)]
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
        for dx in range(size_x):
            for dy in range(size_y):
                for dz in range(size_z):
                    self.used[x+dx][y+dy][z+dz] = True
        for dx in range(size_x):
            for dz in range(size_z):
                self.depth[x+dx][z+dz] = max(self.depth[x+dx][z+dz], y+size_y)

    def unload_box_at(self, position, size):
        x, y, z = position
        size_x, size_y, size_z = size
        for dx in range(size_x):
            for dy in range(size_y):
                for dz in range(size_z):
                    self.used[x+dx][y+dy][z+dz] = False
        for dx in range(size_x):
            for dz in range(size_z):
                for depth_y in range(self.Y-1, -1, -1):
                    if self.used[x+dx][depth_y][z+dz]:
                        self.depth[x+dx][z+dz] = depth_y+1
                        break
                else:
                    self.depth[x+dx][z+dz] = 0

    def print_depth(self):
        # self.depth = [[0] * self.Z for _ in range(self.X)]
        # for x in range(self.X):
        #     for z in range(self.Z):
        #         for y in range(self.Y-1, -1, -1):
        #             if self.used[x][y][z]:
        #                 self.depth[x][z] = y+1
        #                 break

        for z in range(self.Z - 1, -1, -1):
            for x in range(self.X):
                print(f'{self.depth[x][z]:>4d}', end='')
            print()
        # return self.depth

    def calc_possible_volume(self):
        volume = [[self.Y] * self.Z for _ in range(self.X)]
        for x in range(self.X):
            volume[x][self.Z-1] = self.depth[x][self.Z-1]
            for z in range(self.Z-2, -1, -1):
                volume[x][z] = max(self.depth[x][z], volume[x][z+1])
        size_x = 3
        size_y = 3
        size_z = 3
        for x in range(self.X-size_x+1):
            for z in range(self.Z-size_z+1):
                d = max([max(i[z:z+size_z]) for i in volume[x:x+size_x]])
                for dx in range(size_x):
                    for dz in range(size_z):
                        volume[x+dx][z+dz] = min(volume[x+dx][z+dz], d)
                if volume[x][z] > self.Y - size_y: volume[x][z] = self.Y

        return self.total_volume - 1000 * sum([sum(i) for i in volume])

    def calc_empty_volume(self):
        volume = 0
        for x in range(self.X):
            for y in range(self.Y):
                for z in range(self.Z):
                    if not self.used[x][y][z]: volume += 1000

        return volume

    def calc_filled_volume(self):
        volume = 0
        for x in range(self.X):
            for y in range(self.Y):
                for z in range(self.Z):
                    if self.used[x][y][z]: volume += 1000

        return volume

    def load_box_greedy(self, boxes):
        def get_possible_positions(size):
            size_x, size_y, size_z = size
            positions = []
            for x in range(self.X-size_x+1):
                for z in range(self.Z-size_z+1):
                    y = max([max(i[z:z+size_z]) for i in self.depth[x:x+size_x]])
                    if y + size_y >= self.Y: continue
                    if z == 0:
                        positions.append((x, y, z))
                    else:
                        for dx in range(size_x):
                            for dy in range(size_y):
                                if self.used[x+dx][y+dy][z-1]: break
                            else: continue

                            positions.append((x, y, z))
                            break

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
            for size in get_possible_orientations(info):
                positions = get_possible_positions(size)
                if len(positions) == 0: continue
                for position in positions:
                    self.load_box_at(position, size)
                    possible_volume = self.calc_possible_volume()
                    self.unload_box_at(position, size)
                    if possible_volume > best_fit_possible_volume:
                        best_fit_size = size
                        best_fit_position = position
                        best_fit_possible_volume = possible_volume
            if best_fit_position != None:
                self.load_box_at(best_fit_position, best_fit_size)
                self.box_list.append((best_fit_position, best_fit_size))
                self.box_num += 1

                possible_volume_data.append(best_fit_possible_volume)
                empty_volume_data.append(self.calc_empty_volume())
            else: break

        return possible_volume_data, empty_volume_data

def solve_VRP():
    vehicles = []

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

def main():
    def handle(event):
        nonlocal index
        if event.key == 'right':
            index = min(index+1, v.box_num)
        elif event.key == 'left':
            index = max(index-1, 0)
        print(index)
        viewer.update(v.box_list[:index])

    depot, destinations = read_map()
    OD_matrix = read_OD_matrix()
    
    print(depot)
    print(destinations['D_00001'])
    print(OD_matrix['D_00001']['Depot'])
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

    index = v.box_num
    viewer = visualize.box_viewer_3d(handle)
    viewer.update(v.box_list)
    viewer.show()


if __name__ == '__main__':
    main()

# [2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0, 2, 2, 2, 0, 1, 0, 0, 2, 2, 2, 0, 0, 2, 0, 1, 0, 0, 2, 2, 2, 1, 0, 2, 2, 1, 1, 2, 2, 2, 0, 2, 1, 1, 1, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 1, 0, 0, 1, 0, 0, 2, 1, 1, 1, 1, 2, 0, 2, 2, 1, 2, 0, 0, 1, 2, 0, 1, 0, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2]
# [2, 2, 1, 0, 2, 0, 2, 0, 1, 0, 0, 2, 0, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 1, 1, 2, 2, 0, 2, 2, 0, 0, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 2, 1, 1, 2, 0, 1, 2, 1, 0, 0, 1, 2, 2, 0, 0, 1, 0]

# 잘 쌓은 예시
# [0, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2, 1, 1, 2, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 0, 1, 2, 0, 2, 2, 1, 0, 1, 1, 0, 1, 1, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 0, 1, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 0, 2, 0]
