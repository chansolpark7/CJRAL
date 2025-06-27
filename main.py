import json
from collections import defaultdict
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import random

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
    def __init__(self, route, OD_matrix):
        self.route = route
        self.cost = 0
        self.box_list = []

        self.calculate_cost(OD_matrix)

    def calculate_cost(self, OD_matrix):
        length = len(self.route)
        for i in range(length-1):
            start = self.route[i]
            end = self.route[i+1]
            self. cost += OD_matrix[start][end]

    def load_box_greedy(self, boxes):
        # 30x40x30, 30x50x40, 50x60x50cm
        # 160*280*180 x y z
        X = 16
        Y = 28
        Z = 18
        def calc_possible_volume(used):
            pass
        used = [[[False]*18 for _ in range(28)] for _ in range(16)]
        for x in range(1, X-1):
            for z in range(1, Z-1):
                for dx in range(-1, 2):
                    for dz in range(-1, 2):
                        pass
        self.box_list = []

def solve_VRP():
    vehicles = []

def random_boxes(n):
    boxes = []
    for _ in range(n):
        boxes.append(random.randint(0, 2))
    return boxes

def get_possible_orientations(info):
    if info == 0:
        return [30, 30, 40], [30, 40, 30], [40, 30, 30]
    elif info == 1:
        pass
    else:
        pass

def main():
    depot, destinations = read_map()
    OD_matrix = read_OD_matrix()
    
    print(depot)
    print(destinations['D_00001'])
    print(OD_matrix['D_00001']['Depot'])
    vehicles = solve_VRP()


if __name__ == '__main__':
    main()