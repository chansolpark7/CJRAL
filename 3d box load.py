import json
from collections import defaultdict, namedtuple
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import random
import visualize
import time
import numpy

from main import *

class Vehicle2D:
    X = 16
    Y = 28
    total_volume = X*Y*100
    box_sizes= [
        (5, 5),
        (3, 4)
    ]
    box_volumes = [i[0]*i[1] for i in box_sizes]
    TIME_LIMIT = 5

    def __init__(self):
        self.used = [[False]*self.Y for _ in range(self.X)]
        self.depth = [0] * self.X
        self.box_list = []
        self.path = []
        self.box_num = 0
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

    def calc_possible_volume(self, version=2):
        volume = [self.Y]*self.X
        size_x = 3
        size_y = 3
        for x in range(self.X-size_x+1):
            d = max(self.depth[x:x+size_x])
            for dx in range(size_x):
                volume[x+dx] = min(volume[x+dx], d)
            if version == 2 and volume[x] > self.Y - size_y:volume[x] = self.Y

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

    def load_box_bnb(self, boxes_3d):
        # b1 b2 전처리
        b1b2_boxes = [3 if i == 0 else 5 for i in boxes_3d if i != 2]
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
        b1b2_path = solution(b1b2_boxes)
        b3_path = [i//3 for i in range(len(boxes_3d) - len(b1b2_boxes))]

        self.boxes = []
        self.path = []
        b1b2_index = 0
        b3_index = 0
        counter = dict()
        for index, info_3d in enumerate(boxes_3d):
            info_2d = 0 if info_3d == 2 else 1
            if info_2d == 0:
                p = b3_path[b3_index]
                b3_index += 1
            else:
                p = b1b2_path[b1b2_index]
                b1b2_index += 1

            if (info_2d, p) not in counter:
                counter[(info_2d, p)] = len(counter)
                self.boxes.append(info_2d)
            self.path.append(counter[(info_2d, p)])

        # main
        answer_volume = 0
        answer_box_list = None
        self.b1b2_box_num = 0
        self.b3_box_num = 0
        start_t = time.time()
        visited = defaultdict(set)
        loading = []
        d = [[], []]
        for index, info in enumerate(self.boxes): # loading중인 박스 처리 나중에 수정
            loading.append(d[0] + d[1])
            if info == 0:
                if len(d[0]) == 1: d[0].pop(0)
            else:
                if len(d[1]) == 2: d[1].pop(0)
            d[info].append(index)
        print(loading)

        def estimate(index):
            possible_volume = self.calc_possible_volume()
            volume = 0
            for info in self.boxes[index:]:
                if volume + self.box_volumes[info] <= possible_volume:
                    volume += self.box_volumes[info]
                else:
                    break
            return volume
        
        def dfs(index, volume):
            nonlocal answer_volume
            nonlocal answer_box_list

            if time.time() - start_t > self.TIME_LIMIT: return
            if index == len(self.boxes): return

            estimated_volume = estimate(index)
            if volume + estimated_volume <= answer_volume: return

            info = self.boxes[index]
            box_volume = self.box_volumes[info]

            best_fit_boxes = None
            best_fit_possible_volume = -1
            for size in self.get_possible_orientations(info):
                positions = self.get_possible_positions(size)
                for position in positions:
                    self.load_box_at(position, size)
                    self.box_list.append((position, size))
                    possible_volume = self.calc_possible_volume()
                    if best_fit_possible_volume < possible_volume:
                        best_fit_possible_volume = possible_volume
                        best_fit_boxes = [(position, size)]
                    elif best_fit_possible_volume == possible_volume:
                        best_fit_boxes.append((position, size))
                    self.unload_box_at(position, size)
                    self.box_list.pop()
                    if volume + box_volume > answer_volume:
                        answer_volume = volume + box_volume
                        answer_box_list = self.box_list + [(position, size)]
            if best_fit_boxes != None:
                not_shuffling = []
                for position, size in best_fit_boxes:
                    for b_index in loading[index]:
                        b_info = self.boxes[b_index]
                        b_position, b_size = self.box_list[b_index]
                        if info == b_info == 0: continue
                        elif position[0] + size[0] <= b_position[0] or b_position[0] + b_size[0] <= position[0]: continue
                        else: break
                    else:
                        not_shuffling.append((position, size))
                if len(not_shuffling) > 0: best_fit_boxes = not_shuffling # 다른 기준?
                elif index == 0: print(not_shuffling)
                # best_fit_boxes = not_shuffling

                for position, size in best_fit_boxes:
                    if position in visited[size]: continue
                    self.load_box_at(position, size)
                    self.box_list.append((position, size))
                    dfs(index + 1, volume + box_volume)
                    self.unload_box_at(position, size)
                    self.box_list.pop()
                    visited[size].add(position)
                for position, size in best_fit_boxes:
                    visited[size].discard(position)

        dfs(0, 0)
        self.box_list = answer_box_list
        self.box_num = len(self.box_list)
        self.b1b2_box_num = self.boxes[:len(self.box_list)].count(1)
        self.b3_box_num = len(self.box_list) - self.b1b2_box_num
        for position, size in self.box_list:
            self.load_box_at(position, size)

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

    def get_possible_positions(self, size):
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

    def get_possible_orientations(self, info):
        if info == 0:
            return (3, 3, 4), (3, 4, 3), (4, 3, 3)
        elif info == 1:
            return (3, 4, 5), (3, 5, 4), (4, 3, 5), (4, 5, 3), (5, 3, 4), (5, 4, 3)
        else:
            return (5, 5, 6), (5, 6, 5), (6, 5, 5)

    def load_box_greedy(self, boxes):

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
            for size in self.get_possible_orientations(info):
                positions = self.get_possible_positions(size)
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

    def load_box_bnb(self, boxes):
        self.boxes = boxes
        self.box_num = 0
        vehicle_2d = Vehicle2D()
        vehicle_2d.load_box_bnb(boxes)
        height = [0] * vehicle_2d.box_num
        for index, info_3d in enumerate(self.boxes):
            p = vehicle_2d.path[index]
            if p >= vehicle_2d.box_num: continue
            position_2d, size_2d = vehicle_2d.box_list[p]
            position_3d = (*position_2d, height[p])
            size_3d = (*size_2d, [3, 5, 6][info_3d])
            height[p] += size_3d[2]
            self.load_box_at(position_3d, size_3d)
            self.box_list.append((position_3d, size_3d))

def random_boxes(n):
    boxes = []
    for _ in range(n):
        boxes.append(random.randint(0, 2))
    return boxes

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
    # possible_volume_data, empty_volume_data = v.load_box_greedy(boxes)
    v.load_box_bnb(boxes)
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

    # visualize.graph(possible = possible_volume_data, empty = empty_volume_data)

    viewer = visualize.box_viewer_3d(v.box_list)
    viewer.show()


if __name__ == '__main__':
    run()

# [2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0, 2, 2, 2, 0, 1, 0, 0, 2, 2, 2, 0, 0, 2, 0, 1, 0, 0, 2, 2, 2, 1, 0, 2, 2, 1, 1, 2, 2, 2, 0, 2, 1, 1, 1, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 1, 0, 0, 1, 0, 0, 2, 1, 1, 1, 1, 2, 0, 2, 2, 1, 2, 0, 0, 1, 2, 0, 1, 0, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2]
# [2, 2, 1, 0, 2, 0, 2, 0, 1, 0, 0, 2, 0, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 1, 1, 2, 2, 0, 2, 2, 0, 0, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 2, 1, 1, 2, 0, 1, 2, 1, 0, 0, 1, 2, 2, 0, 0, 1, 0]

# 잘 쌓은 예시
# [0, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2, 1, 1, 2, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 0, 1, 2, 0, 2, 2, 1, 0, 1, 1, 0, 1, 1, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 0, 1, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 0, 2, 0]
