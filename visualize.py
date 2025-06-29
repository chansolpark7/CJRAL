import main
import matplotlib.pyplot as plt
import random
import sys
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

# main의 어떤 함수를 이용해서 depo, destinations 정보 불러오는 구조
# main.read_map() -> [depo, destinations]
# - depo = {'longtitude': ~, 'latitude': ~}
# - destinations[destination_id] = {'longtitude': ~, 'latitude': ~}

class box_viewer:
    def __init__(self, callback):
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', callback)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([0, 16])
        self.ax.set_ylim([-28, 0])
        self.ax.set_zlim([0, 18])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.title('3D Box Visualization')
        self.lines = []

    def update(self, box_list):
        for line in self.lines:
            line.remove()
        self.lines = []

        for position, size in box_list:
            x, y, z = position
            dx, dy, dz = size
            # Draw a 3D box as a rectangular prism
            xx = [x, x+dx, x+dx, x, x]
            yy = [-y, -y, -y-dy, -y-dy, -y]
            kwargs = {'alpha': 0.5}
            # Bottom face
            self.lines += self.ax.plot3D(xx, yy, [z]*5, color='b', **kwargs)
            # Top face
            self.lines += self.ax.plot3D(xx, yy, [z+dz]*5, color='r', **kwargs)
            # Vertical lines
            for i in range(4):
                self.lines += self.ax.plot3D([xx[i], xx[i]], [yy[i], yy[i]], [z, z+dz], color='g', **kwargs)
        plt.draw()

    def show(self):
        plt.show()

def histogram(data, start=0, end=1, bins=30):
    plt.hist(data, bins=30, color='skyblue', edgecolor='black', range=(start, end))
    plt.title('Histogram Example')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_vrp():
    depot, destinations = main.read_map()
    vehicles = main.solve_VRP()
    plt.figure(figsize=(10, 8))

    # depot 그리기
    plt.scatter(depot['longitude'], depot['latitude'], c='red', marker='s', s=100, label='Depot')

    # destination 그리기
    for dest_id, pos in destinations.items(): # dest_id에 'D_00001' 이런 거 담기고 pos에 {'longitude': 0, 'latitude': 0} 이런 거 담김
        x = pos['longitude']
        y = pos['latitude']
        plt.scatter(x, y, c='blue')
        plt.text(x + 0.3, y + 0.3, str(dest_id), fontsize=9)

    # 차량 경로 그리기
    # for vehicle in vehicles:
    #     route = vehicle.route
    #     coords = [depot] + [destinations[i] for i in route] + [depot]

    #     x_vals = [pt[0] for pt in coords]
    #     y_vals = [pt[1] for pt in coords]

    #     # 색상과 선 굵기 설정
    #     color = [random.random() for _ in range(3)]  # 랜덤 색
    #     thickness = max(1, 5 - vehicle.capacity_left() // 20)  # 빈공간 20%마다 1씩 줄어듦

    #     plt.plot(x_vals, y_vals, color=color, linewidth=thickness,
    #              label=f'Vehicle {vehicle.id} ({vehicle.capacity_left()}% left)')

    plt.title('Vehicle Routing Problem')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    plot_vrp()