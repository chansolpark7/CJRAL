import json
from collections import defaultdict

class Vehicle:
    def __init__(self, route, OD_matrix):
        self.route = route
        self.cost = 0
        self.box = []

        self.calculate_cost(OD_matrix)

    def calculate_cost(self, OD_matrix):
        length = len(self.route)
        for i in range(length-1):
            start = self.route[i]
            end = self.route[i+1]
            self. cost += OD_matrix[start][end]

    def sort_box(self):
        pass

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

def main():
    depot, destinations = read_map()
    OD_matrix = read_OD_matrix()
    
    print(depot)
    print(destinations['D_00001'])
    print(OD_matrix['D_00001']['Depot'])

if __name__ == '__main__':
    main()