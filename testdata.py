import os
from os.path import basename
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from math import radians, cos, sin, asin, sqrt

def make_id(idx):
    return f"D_{idx:05d}"

def generate_city_network(total_points=500, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed()
        np.random.seed()

    delivery_points = []
    road_lines = []
    current_id = 1

    num_main_cities = max(1, total_points // 300)
    remaining_points = total_points
    main_city_coords = []

    for i in range(num_main_cities):
        cx = random.randint(400, 600)
        cy = random.randint(400, 600)
        main_city_coords.append((cx, cy))

        if i < num_main_cities - 1:
            points_for_this_city = total_points // num_main_cities
        else:
            points_for_this_city = remaining_points

        remaining_points -= points_for_this_city

        n_center = int(points_for_this_city * 0.4)
        n_subcity = int(points_for_this_city * 0.4)
        n_midtown = points_for_this_city - (n_center + n_subcity)

        for _ in range(n_center):
            x = np.random.normal(cx, scale=25)
            y = np.random.normal(cy, scale=25)
            delivery_points.append((make_id(current_id), x, y))
            current_id += 1

        num_subcities = random.randint(3, 6)
        angles = np.linspace(0, 2 * np.pi, num_subcities, endpoint=False)
        random.shuffle(angles)

        subcity_points = n_subcity // num_subcities
        midtown_points = n_midtown // num_subcities

        for theta in angles:
            dist = random.randint(200, 350)
            sub_x = cx + dist * np.cos(theta)
            sub_y = cy + dist * np.sin(theta)
            road_lines.append(((cx, cy), (sub_x, sub_y)))

            for _ in range(subcity_points):
                x = np.random.normal(sub_x, scale=20)
                y = np.random.normal(sub_y, scale=20)
                delivery_points.append((make_id(current_id), x, y))
                current_id += 1

            num_midtowns = random.randint(1, 2)
            for i in range(1, num_midtowns + 1):
                mid_x = cx + (dist * i / (num_midtowns + 1)) * np.cos(theta) + np.random.normal(0, 10)
                mid_y = cy + (dist * i / (num_midtowns + 1)) * np.sin(theta) + np.random.normal(0, 10)
                for _ in range(midtown_points // num_midtowns):
                    x = np.random.normal(mid_x, scale=15)
                    y = np.random.normal(mid_y, scale=15)
                    delivery_points.append((make_id(current_id), x, y))
                    current_id += 1

    while len(delivery_points) < total_points:
        x = random.uniform(300, 700)
        y = random.uniform(300, 700)
        delivery_points.append((make_id(current_id), x, y))
        current_id += 1

    return delivery_points, main_city_coords, road_lines


def convert_to_json(delivery_points, filename):
    base_x, base_y = 500, 500
    base_lon, base_lat = 129.0750875, 35.17982005
    scale = 0.0002

    destinations = []
    for dest_id, x, y in delivery_points:
        lon = base_lon + (x - base_x) * scale
        lat = base_lat + (y - base_y) * scale
        destinations.append({
            "destination_id": dest_id,
            "location": {
                "longitude": round(lon, 6),
                "latitude": round(lat, 6)
            }
        })

    box_sizes = [
        {"width": 30, "length": 40, "height": 30},
        {"width": 50, "length": 60, "height": 50},
        {"width": 30, "length": 50, "height": 40}
    ]

    orders = []
    order_number = 1
    box_number = 1
    for dest in destinations:
        num_boxes = random.randint(1, 5)
        for _ in range(num_boxes):
            size = random.choice(box_sizes)
            orders.append({
                "order_number": order_number,
                "box_id": f"B_{box_number:05d}",
                "destination": dest["destination_id"],
                "dimension": size
            })
            order_number += 1
            box_number += 1

    data = {
        "depot": {
            "destination": "Depot",
            "location": {
                "longitude": base_lon,
                "latitude": base_lat
            },
            "dimension": {
                "width": 0.0,
                "length": 0.0,
                "height": 0.0
            }
        },
        "destinations": destinations,
        "orders": orders
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    distance_km = R * c
    return distance_km * 1000


def export_distance_table(delivery_points, filename):
    base_x, base_y = 500, 500
    base_lon, base_lat = 129.0750875, 35.17982005
    scale = 0.0002
    destinations = []
    for dest_id, x, y in delivery_points:
        lon = base_lon + (x - base_x) * scale
        lat = base_lat + (y - base_y) * scale
        destinations.append({
            "id": dest_id,
            "lon": lon,
            "lat": lat
        })

    header = f"{'ORIGIN':<10}\t{'DESTINATION':<12}\t{'TIME_MIN':>10}\t{'DISTANCE_METER':>15}\n"
    lines = [header]
    avg_speed_kmph = 25

    for i in range(len(destinations)):
        for j in range(len(destinations)):
            if i == j:
                continue
            origin = destinations[i]
            dest = destinations[j]
            dist = haversine(origin["lon"], origin["lat"], dest["lon"], dest["lat"]) * (random.random() + 1)
            time_min = dist / (avg_speed_kmph * 1000 / 60)
            lines.append(
                f"{origin['id']:<10}\t"
                f"{dest['id']:<12}\t"
                f"{time_min:10.3f}\t"
                f"{int(round(dist)):15d}\n"
            )

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)

if __name__ == "__main__":
    assert basename(os.getcwd()) == 'routing'
    os.makedirs("data", exist_ok=True)

    place_list = [random.randint(200, 500) for _ in range(10)]
    print(f"장소 수 리스트: {place_list}")

    for idx, total_delivery_points in enumerate(place_list):
        points, mains, roads = generate_city_network(total_delivery_points, seed=None)

        json_filename = f"data/additional_data_{idx+1:02d}.json"
        txt_filename = f"data/additional_distance_data_{idx+1:02d}.txt"

        convert_to_json(points, filename=json_filename)
        export_distance_table(points, filename=txt_filename)

        print(f"[✅] Saved: {json_filename}")
        print(f"[✅] Saved: {txt_filename}")
