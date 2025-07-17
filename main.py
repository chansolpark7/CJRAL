import openpyxl
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

import time
import math
import numpy
import json
import sys
import copy
import random
from collections import defaultdict, namedtuple, deque

Point = namedtuple('Point', ['longitude', 'latitude'])
Order = namedtuple('Order', ['order_num', 'box_id', 'destination', 'info'])
# Distance = namedtuple('Distance', ['time', 'meter'])

DEBUG = True
LOCAL_SEARCH_DEPTH_LIMIT = 10
INTERNAL_OPTIMIZATION_THRESHOLD = 0.01

if DEBUG:
    import visualize

    CVRP_TIME_LIMIT = 10
    BOX_LOAD_TIME_LIMIT = 5
    LOCAL_SEARCH_TIME_LIMIT = 30
else:
    CVRP_TIME_LIMIT = 10*60
    BOX_LOAD_TIME_LIMIT = 5
    LOCAL_SEARCH_TIME_LIMIT = 60 #

box_size = [
    [30, 40, 30],
    [30, 50, 40],
    [50, 60, 50]
]

box_volume = [i[0]*i[1]*i[2] for i in box_size]

def read_map(filename='Data_Set.json'):
    with open(filename, 'rt', encoding='utf-8') as file:
        raw_data = json.load(file)

    destinations = dict()
    name_to_index = dict()
    index_to_name = []
    destinations['Depot'] = Point(**raw_data['depot']['location'])
    name_to_index['Depot'] = 0
    index_to_name.append('Depot')
    for i, destination in enumerate(raw_data['destinations'], 1):
        destination_id, destination_location = destination['destination_id'], destination['location']
        destinations[destination_id] = Point(**destination_location)
        name_to_index[destination_id] = i
        index_to_name.append(destination_id)

    return destinations, name_to_index, index_to_name

def read_OD_matrix(n, name_to_index, filename='distance-data.txt'):
    with open(filename, 'rt', encoding='utf-8') as file:
        file.readline()
        OD_matrix = [[0]*n for _ in range(n)]
        while True:
            string = file.readline().rstrip()
            if not string: break
            origin, destination, time_min, distance_meter = string.split()
            time_min = float(time_min)
            distance_meter = int(distance_meter)

            i = name_to_index[origin]
            j = name_to_index[destination]
            OD_matrix[i][j] = distance_meter

    return OD_matrix

def read_orders(n, name_to_index, filename='Data_Set.json'):
    with open(filename, 'rt', encoding='utf-8') as file:
        raw_data = json.load(file)

    orders = [[] for _ in range(n)]
    for data in raw_data['orders']:
        info = [100, 120, 160].index(sum(data['dimension'].values()))
        order = Order(data['order_number'], data['box_id'], data['destination'], info)
        index = name_to_index[order.destination]
        orders[index].append(order)

    return orders

class Vehicle2D:
    X = 16
    Y = 28
    total_volume = X*Y*100
    box_sizes= [
        (5, 5),
        (3, 4)
    ]
    box_volumes = [i[0]*i[1] for i in box_sizes]

    def __init__(self):
        self.used = [[False]*self.Y for _ in range(self.X)]
        self.depth = [0] * self.X
        self.box_informations = []
        self.box_num = 0
        self.loaded_box_position_size = []
        self.loaded_box_num = 0
        self.placement_index = []
        self.b1b2_box_num = 0
        self.b3_box_num = 0

    def load_box_at(self, position, size):
        x, y = position
        size_x, size_y = size
        for dx in range(size_x):
            for dy in range(size_y):
                self.used[x+dx][y+dy] = True
                self.depth[x+dx] = y+size_y

    def unload_box_at(self, position, size):
        x, y = position
        size_x, size_y = size
        for dx in range(size_x):
            for dy in range(size_y):
                self.used[x+dx][y+dy] = False
        for dx in range(size_x):
            for depth_y in range(self.Y-1, -1, -1):
                if self.used[x+dx][depth_y]:
                    self.depth[x+dx] = depth_y+1
                    break
            else:
                self.depth[x+dx] = 0

    def calc_possible_volume(self):
        volume = [self.Y]*self.X
        size_x = 3
        size_y = 3
        for x in range(self.X-size_x+1):
            d = max(self.depth[x:x+size_x])
            for dx in range(size_x):
                volume[x+dx] = min(volume[x+dx], d)
            if volume[x] > self.Y - size_y:volume[x] = self.Y

        return self.total_volume - 100 * sum(volume)

    def calc_empty_volume(self):
        volume = 0
        for x in range(self.X):
            for y in range(self.Y):
                if not self.used[x][y]: volume += 100

        return volume

    def calc_filled_volume(self):
        volume = 0
        for x in range(self.X):
            for y in range(self.Y):
                if self.used[x][y]: volume += 100

        return volume

    def get_possible_positions(self, size):
        size_x, size_y = size
        positions = []
        for x in range(self.X-size_x+1):
            y = max(self.depth[x:x+size_x])
            if y + size_y > self.Y: continue
            positions.append((x, y))

        return positions

    def get_possible_orientations(self, info):
        if info == 0:
            return (5, 5),
        else:
            return (3, 4), (4, 3)

    def load_box_greedy(self, box_informations):
        # 30x40x30, 30x50x40, 50x60x50cm
        # 160*280*180 x y z

        self.box_informations = box_informations
        self.box_num = len(self.box_informations)
        self.b1b2_box_num = 0
        self.b3_box_num = 0
        for info in self.box_informations:
            best_fit_position = None
            best_fit_size = None
            best_fit_possible_volume = -1
            for size in self.get_possible_orientations(info):
                positions = self.get_possible_positions(size)
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
                self.loaded_box_position_size.append((best_fit_position, best_fit_size))
                self.box_num += 1
            else: break

    def load_box_bnb(self, box_informations):
        self.box_informations = box_informations
        self.box_num = len(self.box_informations)
        answer_volume = 0
        answer_loaded_box_position_size = None
        self.b1b2_box_num = 0
        self.b3_box_num = 0
        start_t = time.time()
        visited = defaultdict(set)
        loading = []
        loading_queue = [deque(), deque()]
        for index, info in enumerate(self.box_informations): # loading중인 박스 처리 나중에 수정
            loading.append(loading_queue[0] + loading_queue[1])
            if info == 0:
                if len(loading_queue[0]) == 1: loading_queue[0].popleft()
            else:
                if len(loading_queue[1]) == 2: loading_queue[1].popleft()
            loading_queue[info].append(index)

        def estimate(index):
            possible_volume = self.calc_possible_volume()
            volume = 0
            for info in self.box_informations[index:]:
                if volume + self.box_volumes[info] <= possible_volume:
                    volume += self.box_volumes[info]
                else:
                    break
            return volume
        
        def dfs(index, volume):
            nonlocal answer_volume
            nonlocal answer_loaded_box_position_size

            if time.time() - start_t > BOX_LOAD_TIME_LIMIT: return
            if index == len(self.box_informations): return

            estimated_volume = estimate(index)
            if volume + estimated_volume <= answer_volume: return

            info = self.box_informations[index]
            box_volume = self.box_volumes[info]

            best_fit_possible_volume = -1
            best_fit_box_positions = None
            best_fit_box_sizes = None
            for size in self.get_possible_orientations(info):
                positions = self.get_possible_positions(size)
                for position in positions:
                    self.load_box_at(position, size)
                    self.loaded_box_position_size.append((position, size))
                    possible_volume = self.calc_possible_volume()
                    if best_fit_possible_volume < possible_volume:
                        best_fit_possible_volume = possible_volume
                        best_fit_box_positions = [position]
                        best_fit_box_sizes = [size]
                    elif best_fit_possible_volume == possible_volume:
                        best_fit_box_positions.append(position)
                        best_fit_box_sizes.append(size)
                    self.unload_box_at(position, size)
                    self.loaded_box_position_size.pop()
                    if volume + box_volume > answer_volume:
                        answer_volume = volume + box_volume
                        answer_loaded_box_position_size = self.loaded_box_position_size + [(position, size)]
            if best_fit_possible_volume != -1:
                not_shuffling_positions = []
                not_shuffling_sizes = []
                for position, size in zip(best_fit_box_positions, best_fit_box_sizes):
                    for b_index in loading[index]:
                        b_info = self.box_informations[b_index]
                        b_position, b_size = self.loaded_box_position_size[b_index]
                        if info == b_info == 0: continue
                        elif position[0] + size[0] <= b_position[0] or b_position[0] + b_size[0] <= position[0]: continue
                        else: break
                    else:
                        not_shuffling_positions.append(position)
                        not_shuffling_sizes.append(size)
                if len(not_shuffling_positions) > 0: # 다른 기준?
                    best_fit_box_positions = not_shuffling_positions
                    best_fit_box_sizes = not_shuffling_sizes

                for position, size in zip(best_fit_box_positions, best_fit_box_sizes):
                    if position in visited[size]: continue
                    self.load_box_at(position, size)
                    self.loaded_box_position_size.append((position, size))
                    dfs(index + 1, volume + box_volume)
                    self.unload_box_at(position, size)
                    self.loaded_box_position_size.pop()
                    visited[size].add(position)
                for position, size in zip(best_fit_box_positions, best_fit_box_sizes):
                    visited[size].discard(position)

        dfs(0, 0)
        self.loaded_box_position_size = answer_loaded_box_position_size
        self.loaded_box_num = len(self.loaded_box_position_size)
        self.b1b2_box_num = self.box_informations[:len(self.loaded_box_position_size)].count(1)
        self.b3_box_num = len(self.loaded_box_position_size) - self.b1b2_box_num
        for position, size in self.loaded_box_position_size:
            self.load_box_at(position, size)

class Vehicle:
    X = 16
    Y = 28
    Z = 18
    total_volume = X*Y*Z*1000

    def __init__(self, route, orders):
        self.route = route
        self.dist = 0
        self.used = [[[False]*self.Z for _ in range(self.Y)] for _ in range(self.X)]
        self.depth = [[0] * self.Z for _ in range(self.X)]

        self.data_empty_volume = []
        self.data_possible_volume = []

        self.box_informations = []
        self.box_route_index = []
        for index, node in enumerate(self.route):
            for box in orders[node]:
                self.box_informations.append(box.info)
                self.box_route_index.append(index)
        self.box_num = len(self.box_informations)
        self.box_informations.reverse()
        self.box_route_index.reverse()

        self.loaded_box_position_size = []
        self.loaded_box_num = 0
        self.unloaded_route = []

    def calculate_dist(self, OD_matrix):
        length = len(self.route)
        self.dist = 0
        for i in range(length-1):
            start = self.route[i]
            end = self.route[i+1]
            self.dist += OD_matrix[start][end]
        return self.dist

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
        for z in range(self.Z - 1, -1, -1):
            for x in range(self.X):
                print(f'{self.depth[x][z]:>4d}', end='')
            print()

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

    def calc_load_factor(self):
        return self.calc_filled_volume() / self.total_volume

    def get_possible_positions(self, size):
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

    def get_possible_orientations(self, info):
        if info == 0:
            return (3, 3, 4), (3, 4, 3), (4, 3, 3)
        elif info == 1:
            return (3, 4, 5), (3, 5, 4), (4, 3, 5), (4, 5, 3), (5, 3, 4), (5, 4, 3)
        else:
            return (5, 5, 6), (5, 6, 5), (6, 5, 5)

    def load_box_greedy(self, box_informations=None):
        # 30x40x30, 30x50x40, 50x60x50cm
        # 160*280*180 x y z

        if box_informations != None:
            self.box_informations = box_informations
            self.box_num = len(self.box_informations)
        self.data_possible_volume = []
        self.data_empty_volume = []

        self.loaded_box_num = 0
        for info in self.box_informations:
            best_fit_position = None
            best_fit_size = None
            best_fit_possible_volume = -1
            for size in self.get_possible_orientations(info):
                positions = self.get_possible_positions(size)
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
                self.loaded_box_position_size.append((best_fit_position, best_fit_size))
                self.loaded_box_num += 1

                self.data_possible_volume.append(best_fit_possible_volume)
                self.data_empty_volume.append(self.calc_empty_volume())
            else: break

    def load_box_bnb(self, box_informations=None):
        if box_informations != None:
            self.box_informations = box_informations
            self.box_num = len(self.box_informations)

        # b1 b2 전처리
        b1b2_boxes = [3 if i == 0 else 5 for i in self.box_informations if i != 2]
        def solution(boxes):
            n = len(boxes)
            INF = 300

            dp = [[[INF]*19 for _ in range(19)] for _ in range(n + 1)]
            prev = [[[None]*19 for _ in range(19)] for _ in range(n + 1)]
            dp[0][0][0] = 0

            for index in range(n):
                size = boxes[index]
                for i in range(19):
                    for j in range(19):
                        if dp[index][i][j] == INF: continue
                        if i + size > 18:
                            if dp[index + 1][j][size] > dp[index][i][j] + 1:
                                dp[index + 1][j][size] = dp[index][i][j] + 1
                                prev[index + 1][j][size] = (i, j)
                        else:
                            if dp[index + 1][i + size][j] > dp[index][i][j] + (i == 0):
                                dp[index + 1][i + size][j] = dp[index][i][j] + (i == 0)
                                prev[index + 1][i + size][j] = (i, j)
                        if j + size > 18:
                            pass
                        else:
                            if dp[index + 1][i][j + size] > dp[index][i][j] + (j == 0):
                                dp[index + 1][i][j + size] = dp[index][i][j] + (j == 0)
                                prev[index + 1][i][j + size] = (i, j)

            answer = min(min(arr) for arr in dp[n])

            best_i, best_j = None, None
            best_score = -1
            for i in range(19):
                for j in range(19):
                    if dp[n][i][j] == answer:
                        score = (18 - i if i <= 15 else 0) + (18 - j if j <= 15 else 0)
                        if score > best_score:
                            best_i, best_j = i, j
                            best_score = score

            path = []
            box_index = answer-1
            i, j = best_i, best_j
            for index in range(n, 0, -1):
                size = boxes[index-1]
                prev_i, prev_j = prev[index][i][j]
                if prev_i == i and prev_j + size == j:
                    path.append(box_index)
                else:
                    if prev_i + size > 18:
                        path.append(box_index)
                        box_index -= 1
                    else:
                        path.append(box_index-1)
                i, j = prev_i, prev_j
            path.reverse()
            return path
        b1b2_placement_index = solution(b1b2_boxes)
        b3_placement_index = [i//3 for i in range(len(self.box_informations) - len(b1b2_boxes))]

        box_informations_2d = []
        placement_index = []
        b1b2_index = 0
        b3_index = 0
        counter = dict()
        for index, info_3d in enumerate(self.box_informations):
            info_2d = 0 if info_3d == 2 else 1
            if info_2d == 0:
                p = b3_placement_index[b3_index]
                b3_index += 1
            else:
                p = b1b2_placement_index[b1b2_index]
                b1b2_index += 1

            if (info_2d, p) not in counter:
                counter[(info_2d, p)] = len(counter)
                box_informations_2d.append(info_2d)
            placement_index.append(counter[(info_2d, p)])

        # main
        vehicle_2d = Vehicle2D()
        vehicle_2d.load_box_bnb(box_informations_2d)

        # unloaded_route_index 찾기
        if vehicle_2d.loaded_box_num != vehicle_2d.box_num:
            p = placement_index.index(vehicle_2d.loaded_box_num)
            unloaded_route_index = self.box_route_index[p]
        else:
            unloaded_route_index = 0

        height = [0] * vehicle_2d.loaded_box_num
        self.data_possible_volume = [self.total_volume]
        self.data_empty_volume = [self.total_volume]
        for index, info_3d in enumerate(self.box_informations):
            if self.box_route_index[index] <= unloaded_route_index: break
            p = placement_index[index]
            position_2d, size_2d = vehicle_2d.loaded_box_position_size[p]
            position_3d = (*position_2d, height[p])
            size_3d = (*size_2d, [3, 5, 6][info_3d])
            height[p] += size_3d[2]
            self.load_box_at(position_3d, size_3d)
            self.data_possible_volume.append(self.calc_possible_volume())
            self.data_empty_volume.append(self.calc_empty_volume())
            self.loaded_box_position_size.append((position_3d, size_3d))
        self.loaded_box_num = len(self.loaded_box_position_size)

        # 싣지 못한 목적지 unloaded_route에 저장, route, box_num 정리
        if unloaded_route_index != 0:
            self.unloaded_route = self.route[1:unloaded_route_index+1]
            self.route = [self.route[0]] + self.route[unloaded_route_index+1:]
            self.box_informations = self.box_informations[:self.loaded_box_num]
            self.box_route_index = self.box_route_index[:self.loaded_box_num]
            self.box_num = self.loaded_box_num

def feasible_solution_local_search(original_vehicles: list[Vehicle], OD_matrix, orders): # LS
    start_t = time.time()

    def internal_optimization(vehicle: Vehicle):
        for box_index in range(vehicle.loaded_box_num-1, -1, -1):
            if (vehicle.data_empty_volume[box_index+1] - vehicle.data_possible_volume[box_index+1]) / vehicle.total_volume < INTERNAL_OPTIMIZATION_THRESHOLD:
                break
        node = vehicle.box_route_index[box_index]
        if node == len(vehicle.route)-2: node -= 1
        new_route = vehicle.route[:]
        new_route[node], new_route[node + 1] = new_route[node + 1], new_route[node]
        
        new_vehicle = Vehicle(new_route, orders)
        return new_vehicle

    def reassign_destination(vehicles: list[Vehicle], target_vehicle_index):
        min_ratio_vehicle_index = None
        min_ratio = 1
        for index, vehicle in enumerate(vehicles):
            if index == target_vehicle_index: continue
            ratio = vehicle.calc_load_factor()
            if ratio < min_ratio:
                min_ratio_vehicle_index = index
                min_ratio = ratio

        if min_ratio_vehicle_index != None:
            v1 = vehicles[target_vehicle_index]
            v2 = vehicles[min_ratio_vehicle_index]
            new_route = [0] + v2.unloaded_route + v1.unloaded_route + v2.route[1:]
            v1.unloaded_route = []
            new_vehicle = Vehicle(new_route, orders)
            new_vehicle.load_box_bnb()
            vehicles[min_ratio_vehicle_index] = new_vehicle

    def dfs(depth):
        nonlocal vehicles
        if depth == LOCAL_SEARCH_DEPTH_LIMIT: return False
        if time.time() - start_t > LOCAL_SEARCH_TIME_LIMIT: return False

        if all(len(vehicle.unloaded_route) == 0 for vehicle in vehicles): return True

        # 0 : 내부 적재 최적화
        # 1 : 목적지 재할당
        queue: list[tuple[float, int, Vehicle]] = []
        for index, vehicle in enumerate(vehicles):
            # 0
            # r = (vehicle.data_empty_volume[-1] - vehicle.data_possible_volume[-1]) / vehicle.total_volume
            # if r > INTERNAL_OPTIMIZATION_THRESHOLD and len(vehicle.route) > 3:
            #     queue.append((r, 0, index))

            # 1
            if len(vehicle.unloaded_route) != 0:
                queue.append((0, 1, index)) ##### 가중치 결정?

        queue.sort(key=lambda x: x[:2], reverse=True)
        print(f'depth : {depth}')
        vehicle_status(vehicles)
        for index, (_, ls_type, vehicle_index) in enumerate(queue):
            if index != len(queue)-1 and random.random() < 0.4: continue
            if ls_type == 0:
                print('internal optimization')
                vehicle = vehicles[vehicle_index]
                new_vehicle = internal_optimization(vehicle)
                print(vehicle.loaded_box_num,  new_vehicle.loaded_box_num)
                vehicles[vehicle_index] = new_vehicle
            else:
                print('reassign destination')
                reassign_destination(vehicles, vehicle_index)
            break

        return dfs(depth + 1)

    vehicles = copy.deepcopy(original_vehicles)
    i = 0
    while time.time() - start_t < LOCAL_SEARCH_TIME_LIMIT:
        print(f'dfs {i}')
        if dfs(0):
            return True, vehicles
        vehicles = copy.deepcopy(original_vehicles)
        i += 1

    return False, original_vehicles

def solve_vrp_with_capacity(matrix, demands, vehicle_capacities, depot=0):
    num_vehicles = len(vehicle_capacities)
    manager = pywrapcp.RoutingIndexManager(len(matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.SetFixedCostOfAllVehicles(150000)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands[node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,
        True,
        'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = CVRP_TIME_LIMIT
    # search_parameters.log_search = True

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, []

    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        routes.append(route)

    return solution, routes

def VRP(n, OD_matrix, orders, min_load_ratio=0.9, max_load_ratio=0.95) -> list[Vehicle]:
    demands = [0] * n
    for i in range(1, n):
        for order in orders[i]:
            demands[i] += box_volume[order.info]
    total_demand = sum(demands)

    print(total_demand / Vehicle.total_volume)
    print(total_demand / (Vehicle.total_volume * min_load_ratio))
    min_vehicle_num = math.ceil(total_demand / Vehicle.total_volume)
    max_vehicle_num = math.ceil(total_demand / (Vehicle.total_volume * min_load_ratio))
    print(f'vehicle num : {min_vehicle_num} ~ {max_vehicle_num}')
    vehicle_count = max_vehicle_num
    vehicle_capacities = [int(Vehicle.total_volume*max_load_ratio)] * vehicle_count

    solution, routes = solve_vrp_with_capacity(OD_matrix, demands, vehicle_capacities, depot=0)

    if not routes:
        print("route not found")
        return

    vehicles = []
    for route in routes:
        if len(route) != 2:
            vehicle = Vehicle(route, orders)
            vehicles.append(vehicle)

    # if DEBUG: print(f'total cost : {total_cost:,}')
    return vehicles

def save(vehicles: list[Vehicle], destinations: dict[str, Point], orders: list[Order], index_to_name):
    wb = openpyxl.Workbook()
    ws = wb.create_sheet()
    ws = wb.active
    ws.append(['Vehicle_ID', 'Route_Order', 'Destination', 'Order_Number', 'Box_ID', 'Stacking_Order', 'Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z', 'Longitude', 'Latitude', 'Box_Width', 'Box_Length', 'Box_Height'])
    for vehicle_id, vehicle in enumerate(vehicles):
        ws.append([vehicle_id, 1, 'Depot'])
        box_index = vehicle.loaded_box_num-1
        route_order = 2
        for route_index in vehicle.route:
            destination_id = index_to_name[route_index]
            destination = destinations[destination_id]
            for order in orders[route_index]:
                box_position, box_size = vehicle.loaded_box_position_size[box_index]
                x, y, z = box_position
                size_x, size_y, size_z = box_size
                ws.append([vehicle_id, route_order, destination_id, order.order_num, order.box_id, box_index+1, x*10, 280-(y+size_y)*10, z*10, destination.longitude, destination.latitude, size_x*10, size_y*10, size_z*10])
                box_index -= 1
                route_order += 1
        ws.append([vehicle_id, route_order, 'Depot'])
    wb.save("Result.xlsx")

def vehicle_status(vehicles: list[Vehicle]):
    print('-'*30)
    print('| status')
    for index, vehicle in enumerate(vehicles):
        print(f'| vehicle {index} : {vehicle.calc_load_factor()}')
        print(f'| {len(vehicle.unloaded_route)} unloaded route num')
        if vehicle.unloaded_route: print(f'| {vehicle.unloaded_route}')
        print('|')
    print('-'*30)

def main(data_filename, distance_filename):
    start_t = time.time()
    destinations, name_to_index, index_to_name = read_map(data_filename)
    n = len(destinations)
    OD_matrix = read_OD_matrix(n, name_to_index, distance_filename)
    orders = read_orders(n, name_to_index, data_filename)
    if DEBUG: print(f'read file time : {time.time() - start_t}\n')

    if DEBUG: print('start VRP')
    vehicles = VRP(n, OD_matrix, orders, 0.90, 0.95)
    if DEBUG: print(f'VRP time : {time.time() - start_t}\n')

    if DEBUG: print('start load box')
    for vehicle in vehicles: vehicle.load_box_bnb()
    if DEBUG: print(f'loaded box : {time.time() - start_t}\n')

    # if DEBUG:
    #     for index, vehicle in enumerate(vehicles):
    #         print(f'Vehicle {index}')
    #         print(f'route : {vehicle.route}')
    #     print()
    
    if DEBUG:
        vehicle_status(vehicles)
        # for vehicle in vehicles:
        #     visualize.graph(possible=vehicle.data_possible_volume, empty=vehicle.data_empty_volume)
        #     viewer = visualize.box_viewer_3d(vehicle.loaded_box_position_size)
        #     viewer.show()

    if DEBUG: print('start local search')
    success, vehicles = feasible_solution_local_search(vehicles, OD_matrix, orders)
    print(f'{success = }')
    if DEBUG: print(f'local search time : {time.time() - start_t}\n')

    if DEBUG:
        vehicle_status(vehicles)
        # for vehicle in vehicles:
        #     visualize.graph(possible=vehicle.data_possible_volume, empty=vehicle.data_empty_volume)
        #     viewer = visualize.box_viewer_3d(vehicle.loaded_box_position_size)
        #     viewer.show()

    save(vehicles, destinations, orders, index_to_name)

# python311 main.py Data_Set.json distance-data.txt
# python311 main.py additional_data.json additional_distance_data.txt
if __name__ == '__main__':
    data_filename, distance_filename = sys.argv[1:]
    main(data_filename, distance_filename)