from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import random
import math

import main

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
    search_parameters.time_limit.seconds = 10

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

def calc_dist(dist_matrix, route):
    dist = 0
    for i in range(len(route)-1):
        dist += dist_matrix[route[i]][route[i+1]]
    return dist

def VRP():
    destinations, name_to_index, index_to_name = main.read_map()
    n = len(destinations)
    OD_matrix = main.read_OD_matrix(n, name_to_index)
    orders = main.read_orders(n, name_to_index)

    demands = [0] * n
    for i in range(1, n):
        for order in orders[i]:
            demands[i] += main.box_volume[order.info]
    total_demand = sum(demands)

    min_load_ratio = 0.90
    min_vehicle_num = math.ceil(total_demand / main.Vehicle.total_volume)
    max_vehicle_num = math.ceil(total_demand / (main.Vehicle.total_volume * min_load_ratio))
    print(f'vehicle num : {min_vehicle_num} ~ {max_vehicle_num}')
    vehicle_count = max_vehicle_num
    vehicle_capacities = [int(main.Vehicle.total_volume*min_load_ratio)] * vehicle_count

    solution, routes = solve_vrp_with_capacity(OD_matrix, demands, vehicle_capacities, depot=0)

    if not routes:
        print("❌ 경로를 찾지 못했습니다.")
        return

    total_cost = 0
    for i, route in enumerate(routes):
        route_names = [index_to_name[idx] for idx in route]
        cost = int(calc_dist(OD_matrix, route)*0.5)
        total_cost += cost
        print(f"🚚 Vehicle {i+1}: Route: {route_names}")
        print(f'{cost=:,}')

    print()
    print(f'total cost : {total_cost + vehicle_count*150000:,}')

if __name__ == "__main__":
    VRP()
