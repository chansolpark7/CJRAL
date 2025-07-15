import main
import visualize
from scipy.spatial import Delaunay

def dist(a, b):
    ax = a.longitude
    ay = a.latitude
    bx = b.longitude
    by = b.latitude

    return ((bx-ax)**2 + (by-ay)**2) ** 0.5

destinations, name_to_index, index_to_name = main.read_map()
n = len(destinations)
OD_matrix = main.read_OD_matrix(n, name_to_index)
orders = main.read_orders(n, name_to_index)

# 유클리드 거리, 이동 길이 비율
dist_sum = 0
meter_sum = 0
all_pos = list(destinations.keys())
print(all_pos)
n = len(all_pos)
for i in range(n):
    a = all_pos[i]
    for j in range(n):
        if i == j: continue
        b = all_pos[j]
        dist_sum += dist(destinations[a], destinations[b])
        meter_sum += OD_matrix[i][j]

print(dist_sum / (n*(n-1)))
print(meter_sum / (n*(n-1)))
r = meter_sum/dist_sum

# 1km 이하 지점 count, 거리 분포 확인
data = []
count = 0
for i in range(n):
    for j in range(n):
        if i == j: continue
        a = all_pos[i]
        b = all_pos[j]
        d1 = OD_matrix[i][j]
        d2 = OD_matrix[j][i]
        if d1 < 3000:
            data.append(abs(d2-d1)/max(d1, d2))
print(count)
print(len(data))
print(data[0])
visualize.histogram(data, 0, 1)

# 들로네 삼각분할
import numpy as np
import matplotlib.pyplot as plt
points = np.array([destinations[i] for i in all_pos])
tri = Delaunay(points)

count = 0
for i, j, k in tri.simplices:
    a, b, c = all_pos[i], all_pos[j], all_pos[k]  # 3개의 점

    # 각 변의 거리 계산 (a-b, b-c, c-a)
    d1 = OD_matrix[i][j]
    d2 = OD_matrix[j][k]
    d3 = OD_matrix[k][i]

    # 하나라도 1000m 미만이면 count 증가
    if d1 < 1000 or d2 < 1000 or d3 < 1000:
        count += 1
    
print(count//2)

plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')
plt.plot(points[:, 0], points[:, 1], 'o', color='red')

for i, (x, y) in enumerate(points):
    plt.text(x, y, str(i), fontsize=4, color='green')

plt.title("Delaunay Triangulation")
plt.gca().set_aspect('equal')
plt.show()

# 박스 비율
# count = [0, 0, 0]
# for destination_order in orders.values():
#     for order in destination_order:
#         count[order.info] += 1

# print(count)
# [151, 161, 125]

# 총 부피
count = [1000/3]*3
total = 0
for i in range(3):
    total += count[i] * [30*40*30, 30*40*50, 50*50*60][i]

vehicle_volume = main.Vehicle.X*main.Vehicle.Y*main.Vehicle.Z*1000

min_car = total/vehicle_volume
max_car = total/(vehicle_volume*0.75)

print(min_car, max_car)