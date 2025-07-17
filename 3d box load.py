import json
from collections import defaultdict, namedtuple, deque
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import random
import visualize
import time
import numpy

from main import *

BOX_LOAD_TIME_LIMIT = 5

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
                self.box_informations.append(box)
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
                if y + size_y > self.Y: continue
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

    def load_box_greedy(self, node, orders):
        # 30x40x30, 30x50x40, 50x60x50cm
        # 160*280*180 x y z
        for index, box in enumerate(orders[node]):
            # info = box.info
            info = box
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
                self.box_num += 1
                self.loaded_box_num += 1

                self.data_possible_volume.append(best_fit_possible_volume)
                self.data_empty_volume.append(self.calc_empty_volume())
            else:
                for i in range(index):
                    position, size = self.loaded_box_position_size.pop()
                    self.unload_box_at(position, size)
                    self.box_num -= 1
                    self.loaded_box_num -= 1

                    self.data_possible_volume.pop()
                    self.data_empty_volume.pop()
                return False
        self.route = self.route[:-1] + [node, 0]
        # self.box_informations = [box.info for box in orders[node]][::-1] + self.box_informations
        self.box_informations = [box for box in orders[node]][::-1] + self.box_informations
        self.box_route_index = [node] * len(orders[node]) + self.box_route_index
        return True

    def load_box_bnb(self):
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


def internal_optimization(vehicle: Vehicle, orders):
    # print(len(vehicle.data_empty_volume), vehicle.loaded_box_num)
    for box_index in range(vehicle.loaded_box_num-1, -1, -1):
        if (vehicle.data_empty_volume[box_index+1] - vehicle.data_possible_volume[box_index+1]) / vehicle.total_volume < INTERNAL_OPTIMIZATION_THRESHOLD:
            break
    print(f'{box_index = }')
    node = vehicle.box_route_index[box_index]
    if node == len(vehicle.route)-2: node -= 1
    new_route = vehicle.route[:]
    # new_route[node], new_route[node + 12] = new_route[node + 12], new_route[node]
    # new_route[node-6:node+6] = new_route[node-6:node+6][::-1]
    print(new_route, node)
    
    new_vehicle = Vehicle(new_route, orders)
    new_vehicle.load_box_bnb()
    return new_vehicle

def reassign_destination(vehicles: list[Vehicle], target_vehicle_index, orders):
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
        new_route = v2.route[:-1] + v1.unloaded_route + [0]
        v1.unloaded_route = []
        new_vehicle = Vehicle(new_route, orders)
        new_vehicle.load_box_bnb()
        vehicles[min_ratio_vehicle_index] = new_vehicle

def apply_greedy_loading(vehicles: list[Vehicle], orders):
    unloaded_route = []
    for vehicle in vehicles:
        unloaded_route += vehicle.unloaded_route
        vehicle.unloaded_route = []

    demands = defaultdict(int)
    for node in unloaded_route:
        for order in orders[node]:
            # demands[node] += box_volume[order.info]
            demands[node] += box_volume[order]

    loading_order = []
    for node, volume in demands.items():
        loading_order.append((volume, node))
    loading_order.sort(reverse=True)

    unloaded_route = []
    for volume, node in loading_order:
        empty_volume_vehicle_list: list[tuple[int, Vehicle]] = []
        for vehicle in vehicles:
            empty_volume_vehicle_list.append((vehicle.calc_empty_volume(), vehicle))
        empty_volume_vehicle_list.sort()

        for empty_volume, vehicle in empty_volume_vehicle_list:
            if empty_volume < volume: continue
            if vehicle.load_box_greedy(node, orders): break
        else:
            unloaded_route.append(node)
    return unloaded_route

def random_boxes(n):
    boxes = []
    for _ in range(n):
        boxes.append(random.randint(0, 2))
        # boxes.append(random.choices([0, 1, 2], weights=[1/5, 2/5, 2/5])[0])
    return boxes

def run():
    # destinations, name_to_index, index_to_name = read_map()
    # n = len(destinations)
    # OD_matrix = read_OD_matrix(n, name_to_index)
    # orders = read_orders(n, name_to_index)
    # vehicles = solve_VRP()

    # t = time.time()



    # n1 = 70
    # n2 = 30
    # route1 = [0] + [i for i in range(1, n1+1)] + [0]
    # route2 = [0] + [i for i in range(n1+1, n1+n2+1)] + [0]
    # orders = [[]] + [random_boxes(random.randint(1, 2)) for i in range(n1+n2)] + [[]]
    # # route = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 0]
    # # orders = [[], [1], [1], [1], [1, 1], [0, 1], [0, 1], [2], [1, 2], [1], [1, 1], [2, 1], [2, 0], [1, 1], [2], [2, 2], [0], [1], [2, 2], [1, 0], [1], [2, 2], [1], [1], [1], [1], [1], [2], [1], [1], [2], [0], [2], [1], [1], [1], [1], [2, 2], [0], [2], [1, 0], [1], [0, 1], [1], [2], [2, 1], [1], [2, 1], [1], [2], [1, 0], [2, 0], [2], [2, 2], [1], [2, 0], [2, 0], [2, 2], [1], [2, 1], [0], [2, 0], [1, 2], [2, 1], [0], [0, 0], [2, 1], [0], [2, 2], [2, 1], [1], [2], [2, 0], [1, 1], [1], [2, 2], [1, 2], [2, 0], [1], [2, 1], [0, 2], [1, 1], [0, 2], [2, 2], [1, 1], [1, 0], [2], [0, 0], [1], [2], [1, 1], [1], [0], [2], [1], [2, 0], [2], [1, 0], [2], [0, 1], [1], [2], [1, 2], [2], [2, 1], [2, 2], [2], [1], [1, 0], [2], [1, 0], [1], [0], [1, 1], [1], [2], [1, 2], [0, 2], [1], [1, 2], [2, 1], [1], [0], [1, 1], [0], [1, 2], [1, 2], [2], [2, 0], [2, 2], [1], [1], [1], [2], [1, 0], [1, 1], [0, 2], [2], [1], [2], [2], [0], [2, 0], [2], [2, 2], [2, 1], [1], [1, 0], [2], [0], [2], [2, 0], [2, 2], [1], [1, 1], [1, 1], [2, 1], [0, 2], [2, 2], [2, 1], [1], [1], [2, 2], [1, 2], [2, 0], [2], [0, 1], [2], [2, 2], [1], [2, 2], [2, 2], [2], [2], [0], [1], [1, 2], [0], [1], [2, 2], [0, 0], [1, 0], [1], [0, 1], [1], [1], [0], [0, 2], [1], [2, 1], [2, 1], [0], [0, 2], [1], [2, 1], [2, 1], [2, 1], [1], [1], [2, 0], [1, 1], []]
    # print(route1, orders)
    # print(route2, orders)
    # print()
    # v1 = Vehicle(route1, orders)
    # v1.load_box_bnb()
    # v2 = Vehicle(route2, orders)
    # v2.load_box_bnb()
    # vehicles = [v1, v2]
    # print('time :', time.time() - t)

    # print(f'{v1.calc_load_factor() = }')
    # print(f'{v2.calc_load_factor() = }')
    # print(v1.unloaded_route)
    # print(v2.unloaded_route)
    # print()
    # visualize.box_viewer_3d(v1.loaded_box_position_size).show()
    # visualize.box_viewer_3d(v2.loaded_box_position_size).show()

    # print('reassign destination')
    # reassign_destination(vehicles, 0, orders)
    # v1, v2 = vehicles

    # print(f'{v1.calc_load_factor() = }')
    # print(f'{v2.calc_load_factor() = }')
    # print(v1.unloaded_route)
    # print(v2.unloaded_route)
    # print()
    # visualize.box_viewer_3d(v1.loaded_box_position_size).show()
    # visualize.box_viewer_3d(v2.loaded_box_position_size).show()


    n = 70
    route = [0] + [i for i in range(1, n+1)] + [0]
    orders = [[]] + [random_boxes(random.randint(1, 2)) for i in range(n)] + [[]]
    # route = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 0]
    # orders = [[], [2], [1], [1], [0, 1], [2, 0], [1], [1, 1], [2], [2], [0, 2], [1, 2], [1], [2, 2], [2], [1], [1, 2], [1], [0], [2, 2], [2, 1], [0, 0], [1], [0, 2], [1], [2], [0], [1], [1, 2], [0], [0], [2], [2, 2], [2, 0], [1, 1], [1, 0], [2, 1], [2, 0], [2, 1], [1, 1], [2], [2], [0], [0], [1, 1], [2], [1, 0], [1, 2], [1, 2], [1], [0], [1], [0, 0], [2, 1], [0, 0], [2, 0], [0], [2, 2], [1, 0], [1], [2, 2], [0], [0, 0], [1], [1], [2, 0], [2, 0], [2], [1], [1, 1], [1], []]
    print(route)
    print(orders)
    v = Vehicle(route, orders)
    v.load_box_bnb()

    print(v.unloaded_route)
    visualize.box_viewer_3d(v.loaded_box_position_size).show()

    if v.unloaded_route:
        unloaded_route = apply_greedy_loading([v], orders)
        print(unloaded_route)

        visualize.box_viewer_3d(v.loaded_box_position_size).show()
    
    # empty = v1.calc_empty_volume()
    # possible = v1.calc_possible_volume()
    
    # v = Vehicle([], [])
    # boxes = random_boxes(100)
    # print(boxes)
    # # v.load_box_greedy(boxes)
    # v.load_box_bnb(boxes)

    # v.print_depth()

    # visualize.graph(possible = v1.data_possible_volume, empty = v1.data_empty_volume)

    # viewer = visualize.box_viewer_3d(v1.loaded_box_position_size)
    # viewer.show()
    # viewer = visualize.box_viewer_3d(new_v.loaded_box_position_size)
    # viewer.show()

if __name__ == '__main__':
    run()