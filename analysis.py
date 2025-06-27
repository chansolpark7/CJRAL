import main
import visualize

def dist(a, b):
    ax = a['longitude']
    ay = a['latitude']
    bx = b['longitude']
    by = b['latitude']

    return ((bx-ax)**2 + (by-ay)**2) ** 0.5

depot, destinations = main.read_map()
OD_matrix = main.read_OD_matrix()

destinations['Depot'] = depot
print(destinations)
dist_sum = 0
meter_sum = 0
all_pos = list(destinations.keys())
n = len(all_pos)
for i in range(n):
    a = all_pos[i]
    for j in range(n):
        if i == j: continue
        b = all_pos[j]
        dist_sum += dist(destinations[a], destinations[b])
        meter_sum += OD_matrix[a][b][1]

print(dist_sum / (n*(n-1)))
print(meter_sum / (n*(n-1)))
r = meter_sum/dist_sum

data = []
for i in range(n):
    for j in range(n):
        if i == j: continue
        a = all_pos[i]
        b = all_pos[j]
        d = dist(destinations[a], destinations[b])
        meter = OD_matrix[a][b][1]
        delta = (d*r - meter) / meter
        data.append(delta)

visualize.histogram(data)