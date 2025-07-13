import main
import matplotlib.pyplot as plt
import random
import sys
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider

# main의 어떤 함수를 이용해서 depo, destinations 정보 불러오는 구조
# main.read_map() -> [depo, destinations]
# - depo = {'longtitude': ~, 'latitude': ~}
# - destinations[destination_id] = {'longtitude': ~, 'latitude': ~}

class box_viewer_3d:
    def __init__(self, box_list, mode=1):
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_callback)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([0, 16])
        self.ax.set_ylim([-28, 0])
        self.ax.set_zlim([0, 18])
        self.ax.set_box_aspect([16, 28, 18])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.title('3D Box Visualization')

        plt.subplots_adjust(bottom=0.25)
        self.slider_ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.03])
        self.slider = Slider(self.slider_ax, 'Time', 0, len(box_list), valinit=len(box_list), valstep=1)
        self.slider.on_changed(self.update)

        self.box_list = box_list
        self.mode = mode
        self.lines = []
        self.artists = []

        self.colors = [[random.random() * 0.7 + 0.3 for _ in range(3)] for _ in range(100)]
        self.update(self.slider.val)

    def key_callback(self, event):
        val = int(self.slider.val)
        if event.key == 'right' and val < self.slider.valmax:
            self.slider.set_val(val + 1)
        elif event.key == 'left' and val > self.slider.valmin:
            self.slider.set_val(val - 1)

    def update(self, value):
        if self.mode == 0: # line
            for line in self.lines:
                line.remove()
            self.lines = []

            for position, size in self.box_list[:value]:
                x, y, z = position
                dx, dy, dz = size
                y, dy = -y, -dy
                # Draw a 3D box as a rectangular prism
                xx = [x, x+dx, x+dx, x, x]
                yy = [y, y, y+dy, y+dy, y]
                kwargs = {'alpha': 0.5}
                # Bottom face
                self.lines += self.ax.plot3D(xx, yy, [z]*5, color='b', **kwargs)
                # Top face
                self.lines += self.ax.plot3D(xx, yy, [z+dz]*5, color='r', **kwargs)
                # Vertical lines
                for i in range(4):
                    self.lines += self.ax.plot3D([xx[i], xx[i]], [yy[i], yy[i]], [z, z+dz], color='g', **kwargs)
        elif self.mode == 1: # face
            for artist in self.artists:
                artist.remove()
            self.artists = []

            for index, (position, size) in enumerate(self.box_list[:value]):
                x, y, z = position
                dx, dy, dz = size
                y, dy = -y, -dy  # y축 반전

                # 8개 꼭짓점
                corners = [
                    [x, y, z],
                    [x + dx, y, z],
                    [x + dx, y + dy, z],
                    [x, y + dy, z],
                    [x, y, z + dz],
                    [x + dx, y, z + dz],
                    [x + dx, y + dy, z + dz],
                    [x, y + dy, z + dz],
                ]

                # 6개 면 (각각 꼭짓점 4개)
                faces = [
                    [corners[0], corners[1], corners[2], corners[3]],  # bottom
                    [corners[4], corners[5], corners[6], corners[7]],  # top
                    [corners[0], corners[1], corners[5], corners[4]],  # front
                    [corners[2], corners[3], corners[7], corners[6]],  # back
                    [corners[1], corners[2], corners[6], corners[5]],  # right
                    [corners[3], corners[0], corners[4], corners[7]],  # left
                ]

                face_color = self.colors[index]
                box = Poly3DCollection(faces, facecolors=face_color, edgecolors='k', linewidths=0.5, alpha=0.5)
                self.ax.add_collection3d(box)
                self.artists.append(box)

        plt.draw()

    def show(self):
        plt.show()

class box_viewer_2d:
    def __init__(self, box_list):
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_callback)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([0, 16])
        self.ax.set_ylim([-28, 0])
        plt.gca().set_aspect('equal', adjustable='box')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        plt.title('2D Box Visualization')

        plt.subplots_adjust(bottom=0.25)
        self.slider_ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.03])
        self.slider = Slider(self.slider_ax, 'Time', 0, len(box_list), valinit=len(box_list), valstep=1)
        self.slider.on_changed(self.update)

        self.box_list = box_list
        self.lines = []

        self.colors = [[random.random() * 0.7 + 0.3 for _ in range(3)] for _ in range(100)]
        self.update(self.slider.val)

    def key_callback(self, event):
        val = int(self.slider.val)
        if event.key == 'right' and val < self.slider.valmax:
            self.slider.set_val(val + 1)
        elif event.key == 'left' and val > self.slider.valmin:
            self.slider.set_val(val - 1)

    def update(self, val):
        for line in self.lines:
            line.remove()
        self.lines = []

        for position, size in self.box_list[:val]:
            x, y = position
            dx, dy = size
            y, dy = -y, -dy
            xx = [x, x+dx, x+dx, x, x]
            yy = [y, y, y+dy, y+dy, y]
            kwargs = {'alpha': 0.5}
            self.lines += self.ax.plot(xx, yy, color='rg'[(10, 7).index(sum(size))], **kwargs)
        plt.draw()

    def show(self):
        plt.show()

def histogram(data, start=0, end=1, bins=30):
    plt.hist(data, bins=30, color='skyblue', edgecolor='black', range=(start, end))
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def graph(**datas):
    for name, data in datas.items():
        plt.plot(data, label=name)
    plt.title('graph')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_vrp():
    # data_file_name = 'Data_Set.json'
    # distance_file_name = 'distance-data.txt'
    data_file_name = 'additional_data.json'
    distance_file_name = 'additional_distance_data.txt'
    destinations, name_to_index, index_to_name = main.read_map(data_file_name)
    n = len(destinations)
    OD_matrix = main.read_OD_matrix(n, name_to_index, distance_file_name)
    orders = main.read_orders(n, name_to_index, data_file_name)
    vehicles = main.VRP(n, OD_matrix, orders)
    plt.figure(figsize=(10, 8))

    # depot 그리기
    plt.scatter(destinations['Depot'].longitude, destinations['Depot'].latitude, c='red', marker='s', s=100, label='Depot')

    # destination 그리기
    for dest_id, pos in list(destinations.items())[1:]: # dest_id에 'D_00001' 이런 거 담기고 pos에 {'longitude': 0, 'latitude': 0} 이런 거 담김
        x = pos.longitude
        y = pos.latitude
        plt.scatter(x, y, c='blue')
        # plt.text(x, y, str(dest_id), fontsize=9)

    # 차량 경로 그리기
    for vehicle in vehicles:
        route = vehicle.route
        x_vals = []
        y_vals = []
        for route_index in vehicle.route:
            destination_id = index_to_name[route_index]
            destination = destinations[destination_id]
            # coords.append((destination.longitude, destination.latitude))
            x_vals.append(destination.longitude)
            y_vals.append(destination.latitude)

        # 색상과 선 굵기 설정
        color = [random.random() for _ in range(3)]  # 랜덤 색
        thickness = max(1, 5 - vehicle.calc_possible_volume()/vehicle.total_volume*5)  # 빈공간 20%마다 1씩 줄어듦

        plt.plot(x_vals, y_vals, color=color, linewidth=thickness,
                 label=f'Vehicle ({vehicle.calc_possible_volume()/vehicle.total_volume*100: .2f}% left)')

    plt.title('Vehicle Routing Problem')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    plot_vrp()