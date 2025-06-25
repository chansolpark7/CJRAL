import main
import pygame
import matplotlib.pyplot as plt
import random
import sys

print(sys.path)
# 여기다 시각화

# main의 어떤 함수를 이용해서 depo, destinations 정보 불러오는 구조
# main.read_map() -> [depo, destinations]
# - depo = {'longtitude': ~, 'latitude': ~}
# - destinations[destination_id] = {'longtitude': ~, 'latitude': ~}

# pygame
# def show():
#     size = (600, 600)

#     pygame.init()
#     screen = pygame.display.set_mode(size)

#     clock = pygame.time.Clock()

#     depot, destinations = main.read_map()

#     while True:
#         clock.tick(60)
print(sys.path)

depot, destinations = main.read_map() 
vehicles = main.solve_VRP()
# print(depot)
# print(destinations)
def plot_vrp(depot, destinations, vehicles):
    plt.figure(figsize=(10, 8))

    # depot 그리기
    print(depot)
    plt.scatter(depot['longitude'], depot['latitude'], c='red', marker='s', s=200, label='Depot')

    # destination 그리기
    for dest_id, (x, y) in destinations.items():
        plt.scatter(x, y, c='blue')
        plt.text(x + 0.3, y + 0.3, str(dest_id), fontsize=9)

    # 차량 경로 그리기
    for vehicle in vehicles:
        route = vehicle.route
        coords = [depot] + [destinations[i] for i in route] + [depot]

        x_vals = [pt[0] for pt in coords]
        y_vals = [pt[1] for pt in coords]

        # 색상과 선 굵기 설정
        color = [random.random() for _ in range(3)]  # 랜덤 색
        thickness = max(1, 5 - vehicle.capacity_left() // 20)  # 빈공간 20%마다 1씩 줄어듦

        plt.plot(x_vals, y_vals, color=color, linewidth=thickness,
                 label=f'Vehicle {vehicle.id} ({vehicle.capacity_left()}% left)')

    plt.title('Vehicle Routing Problem')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
plot_vrp(depot, destinations, vehicles)

class Vehicle:
    def __init__(self, id, capacity, route):
        self.id = id
        self.capacity = capacity
        self.route = route
        self.load = sum(...)  # 목적지 별 택배 무게 계산

    def capacity_left(self):
        return int(100 * (self.capacity - self.load) / self.capacity)
    
