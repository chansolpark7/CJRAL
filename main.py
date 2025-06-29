import json
from collections import defaultdict
# from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import random
import visualize

box_info = [
    [30, 40, 30],
    [30, 50, 40],
    [50, 60, 50]
]

def read_map():
    with open('Data_Set.json', 'rt', encoding='utf-8') as file:
        raw_data = json.load(file)

    depot = raw_data['depot']['location']
    destinations = dict()
    for destination in raw_data['destinations']:
        destinations[destination['destination_id']] = destination['location']

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

    def unload_box_at(self, position, size):
        x, y, z = position
        size_x, size_y, size_z = size
        for dx in range(size_x):
            for dy in range(size_y):
                for dz in range(size_z):
                    self.used[x+dx][y+dy][z+dz] = False

    def get_depth(self):
        depth = [[0] * self.Z for _ in range(self.X)]
        for x in range(self.X):
            for z in range(self.Z):
                for y in range(self.Y-1, -1, -1):
                    if self.used[x][y][z]:
                        depth[x][z] = y+1
                        break

        return depth

    def calc_possible_volume(self):
        volume = [[self.Y] * self.Z for _ in range(self.X)]
        depth = self.get_depth()
        # for x in range(self.X):
        #     for z in range(self.Z-2, -1, -1):
        #         depth[x][z] = max(depth[x][z], depth[x][z+1])
        size_x = 3
        size_z = 3
        for x in range(self.X-size_x+1):
            for z in range(self.Z-size_z+1):
                d = max([max(i[z:z+size_z]) for i in depth[x:x+size_x]])
                for dx in range(size_x):
                    for dz in range(size_z):
                        volume[x+dx][z+dz] = min(volume[x+dx][z+dz], d)

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
            depth = self.get_depth()
            positions = []
            for x in range(self.X-size_x+1):
                for z in range(self.Z-size_z+1):
                    y = max([max(i[z:z+size_z]) for i in depth[x:x+size_x]])
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
            else: break

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
        viewer.update(v.box_list[:index])

    depot, destinations = read_map()
    OD_matrix = read_OD_matrix()
    
    print(depot)
    print(destinations['D_00001'])
    print(OD_matrix['D_00001']['Depot'])
    # vehicles = solve_VRP()

    boxes = random_boxes(100)
    print(boxes)

    v = Vehicle([], OD_matrix)
    v.load_box_greedy(boxes)
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

    index = v.box_num
    viewer = visualize.box_viewer(handle)
    viewer.update(v.box_list)
    viewer.show()


if __name__ == '__main__':
    main()