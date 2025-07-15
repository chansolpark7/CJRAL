import openpyxl

import time
import math
import numpy
import json
import sys
from collections import defaultdict, namedtuple, deque

# import visualize

Point = namedtuple('Point', ['longitude', 'latitude'])
Order = namedtuple('Order', ['order_num', 'box_id', 'destination', 'info'])
# Distance = namedtuple('Distance', ['time', 'meter'])

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

class Vehicle:
    X = 16
    Y = 28
    Z = 18
    total_volume = X*Y*Z*1000

    def __init__(self, route, OD_matrix, orders):
        self.route = route
        self.cost = 0
        self.used = [[[False]*self.Z for _ in range(self.Y)] for _ in range(self.X)]
        self.depth = [[0] * self.Z for _ in range(self.X)]
        self.box_informations = []
        self.box_num = 0
        self.loaded_box_position_size = []
        self.loaded_box_num = 0

        for index in self.route[1:-1]:
            for box in orders[index]:
                self.box_informations.append(box.info)
        self.box_num = len(self.box_informations)
        self.box_informations.reverse()

        self.calculate_dist(OD_matrix)

    def calculate_dist(self, OD_matrix):
        length = len(self.route)
        self.dist = 0
        for i in range(length-1):
            start = self.route[i]
            end = self.route[i+1]
            self.dist += OD_matrix[start][end]

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

    def load_box(self, box_informations=None):
        if box_informations != None:
            self.box_informations = box_informations
            self.box_num = len(self.box_informations)
        
        max_cols = 3
        max_rows = 5
        width_stride = 5
        length_stride = 5
        heights = [[0 for _ in range(max_rows)] for _ in range(max_cols)]

        for index, info in enumerate(self.box_informations):
            index = -index - 1
            col = index % max_cols
            row = (index // max_cols) % max_rows
            position = (col*width_stride, row*length_stride, heights[col][row])
            size = self.get_possible_orientations(info)[0]
            self.loaded_box_position_size.append((position, size))
            heights[col][row] += size[2]
        self.loaded_box_num = len(self.loaded_box_position_size)

def VRP(n, OD_matrix, orders) -> list[Vehicle]:
    vehicles = []
    total_cost = 0
    for i in range(1, n):
        vehicle = Vehicle([0, i, 0], OD_matrix, orders)
        vehicle.load_box()
        vehicles.append(vehicle)
        total_cost += int(vehicle.dist*0.5) + 150000

    print(f'total cost : {total_cost:,}')
    return vehicles

def save(vehicles: list[Vehicle], destinations: dict[str, Point], orders: list[Order], index_to_name):
    wb = openpyxl.Workbook()
    ws = wb.create_sheet()
    ws = wb.active
    ws.append(['Vehicle_ID', 'Route_Order', 'Destination', 'Order_Number', 'Box_ID', 'Stacking_Order', 'Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z', 'Longitude', 'Latitude', 'Box_Width', 'Box_Length', 'Box_Height'])
    for vehicle_id, vehicle in enumerate(vehicles):
        ws.append([vehicle_id, 1, 'Depot'])
        route_order = 2
        box_index = vehicle.loaded_box_num-1
        for route_index in vehicle.route[1:-1]:
            destination_id = index_to_name[route_index]
            destination = destinations[destination_id]
            for order in orders[route_index]:
                box_position, box_size = vehicle.loaded_box_position_size[box_index]
                ws.append([vehicle_id, route_order, destination_id, order.order_num, order.box_id, box_index+1, *map(lambda x: x*10, box_position), destination.longitude, destination.latitude, *map(lambda x: x*10, box_size)])
                route_order += 1
                box_index -= 1
        ws.append([vehicle_id, route_order, 'Depot'])
    wb.save("Result.xlsx")

def main(data_filename, distance_filename):
    destinations, name_to_index, index_to_name = read_map(data_filename)
    n = len(destinations)
    OD_matrix = read_OD_matrix(n, name_to_index, distance_filename)
    orders = read_orders(n, name_to_index, data_filename)

    t = time.time()
    vehicles = VRP(n, OD_matrix, orders)
    print('time :', time.time() - t)

    for index, vehicle in enumerate(vehicles, 1):
        print(f'Vehicle {index}')
        print(vehicle.loaded_box_position_size)
        filled_volume = vehicle.calc_filled_volume()
        print(f'ratio : {filled_volume/vehicle.total_volume}')
        assert vehicle.loaded_box_num == vehicle.box_num, 'load fail'
        print()
        # viewer = visualize.box_viewer_3d(vehicle.loaded_box_position_size)
        # viewer.show()

    save(vehicles, destinations, orders, index_to_name)

# python311 main.py Data_Set.json distance-data.txt
# python311 main.py additional_data.json additional_distance_data.txt
if __name__ == '__main__':
    data_filename, distance_filename = sys.argv[1:]
    main(data_filename, distance_filename)