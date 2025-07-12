import time
import math
import numpy
import json
import sys
import openpyxl
from collections import defaultdict, namedtuple
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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

def get_possible_orientations(info):
    if info == 0:
        return (3, 3, 4), (3, 4, 3), (4, 3, 3)
    elif info == 1:
        return (3, 4, 5), (3, 5, 4), (4, 3, 5), (4, 5, 3), (5, 3, 4), (5, 4, 3)
    else:
        return (5, 5, 6), (5, 6, 5), (6, 5, 5)

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
        self.boxes = []
        self.box_num = 0
        self.placed_boxes = []
        self.placed_box_num = 0

        for index in self.route[1:-1]:
            for box in orders[index]:
                self.boxes.append(box.info)
        self.box_num = len(self.boxes)
        self.boxes.reverse()

        self.calculate_dist(OD_matrix)
        self.load_box_greedy()

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

    def load_box_greedy(self):
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

        self.placed_box_num = 0
        for info in self.boxes:
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
                self.placed_boxes.append((best_fit_position, best_fit_size))
                self.placed_box_num += 1

                possible_volume_data.append(best_fit_possible_volume)
                empty_volume_data.append(self.calc_empty_volume())
            else: break

        return possible_volume_data, empty_volume_data

def solve_vrp_with_capacity(matrix, demands, vehicle_capacities, depot=0):
    num_vehicles = len(vehicle_capacities)
    manager = pywrapcp.RoutingIndexManager(len(matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

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
    search_parameters.time_limit.seconds = 10*60

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

def VRP(n, OD_matrix, orders) -> list[Vehicle]:
    demands = [0] * n
    for i in range(1, n):
        for order in orders[i]:
            demands[i] += box_volume[order.info]
    total_demand = sum(demands)

    min_load_ratio = 0.40
    max_load_ratio = 0.50
    min_vehicle_num = math.ceil(total_demand / Vehicle.total_volume)
    max_vehicle_num = math.ceil(total_demand / (Vehicle.total_volume * min_load_ratio))
    print(f'vehicle num : {min_vehicle_num} ~ {max_vehicle_num}')
    vehicle_count = max_vehicle_num
    vehicle_capacities = [int(Vehicle.total_volume*max_load_ratio)] * vehicle_count

    solution, routes = solve_vrp_with_capacity(OD_matrix, demands, vehicle_capacities, depot=0)

    if not routes:
        print("route not found")
        return

    total_cost = 0
    vehicles = []
    for i, route in enumerate(routes):
        if len(route) != 2:
            vehicle = Vehicle(route, OD_matrix, orders)
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
        box_index = vehicle.placed_box_num-1
        for route_index in vehicle.route[1:-1]:
            destination_id = index_to_name[route_index]
            destination = destinations[destination_id]
            for order in orders[route_index]:
                box_position, box_size = vehicle.placed_boxes[box_index]
                ws.append([vehicle_id, route_order, destination_id, order.order_num, order.box_id, box_index, *map(lambda x: x*10, box_position), destination.longitude, destination.latitude, *map(lambda x: x*10, box_size)])
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
        print(vehicle.boxes)
        filled_volume = vehicle.calc_filled_volume()
        print(f'ratio : {filled_volume/vehicle.total_volume}')
        print(vehicle.box_num, vehicle.placed_box_num)
        assert vehicle.box_num == vehicle.placed_box_num, 'load fail'
        print()
        # viewer = visualize.box_viewer_3d(vehicle.placed_boxes)
        # viewer.show()

    save(vehicles, destinations, orders, index_to_name)

# python311 main.py Data_Set.json distance-data.txt
if __name__ == '__main__':
    data_filename, distance_filename = sys.argv[1:]
    main(data_filename, distance_filename)