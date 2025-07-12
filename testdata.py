import numpy as np
import matplotlib.pyplot as plt
import random
import json

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

        # 남은 포인트 중 일부만 할당
        if i < num_main_cities - 1:
            points_for_this_city = total_points // num_main_cities
        else:
            # 마지막 도시에는 남은 모든 포인트 몰아주기
            points_for_this_city = remaining_points

        remaining_points -= points_for_this_city

        n_center = int(points_for_this_city * 0.4)
        n_subcity = int(points_for_this_city * 0.4)
        n_midtown = points_for_this_city - (n_center + n_subcity)

        # 중심 도시 클러스터
        for _ in range(n_center):
            x = np.random.normal(cx, scale=25)
            y = np.random.normal(cy, scale=25)
            delivery_points.append((make_id(current_id), x, y))
            current_id += 1

        # 소도시
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

            # 소도시 클러스터
            for _ in range(subcity_points):
                x = np.random.normal(sub_x, scale=20)
                y = np.random.normal(sub_y, scale=20)
                delivery_points.append((make_id(current_id), x, y))
                current_id += 1

            # 도로 중간도시 클러스터
            num_midtowns = random.randint(1, 2)
            for i in range(1, num_midtowns + 1):
                mid_x = cx + (dist * i / (num_midtowns + 1)) * np.cos(theta) + np.random.normal(0, 10)
                mid_y = cy + (dist * i / (num_midtowns + 1)) * np.sin(theta) + np.random.normal(0, 10)
                for _ in range(midtown_points // num_midtowns):
                    x = np.random.normal(mid_x, scale=15)
                    y = np.random.normal(mid_y, scale=15)
                    delivery_points.append((make_id(current_id), x, y))
                    current_id += 1

    # 부족한 개수 채우기
    while len(delivery_points) < total_points:
        x = random.uniform(300, 700)
        y = random.uniform(300, 700)
        delivery_points.append((make_id(current_id), x, y))
        current_id += 1

    return delivery_points, main_city_coords, road_lines

def plot_city_network(delivery_points, road_lines):
    xs = [p[1] for p in delivery_points]
    ys = [p[2] for p in delivery_points]

    plt.figure(figsize=(12, 10))
    plt.scatter(xs, ys, s=40, c='skyblue', edgecolors='k', label="Delivery Point")

    # 출발지(Depot)
    depot_x, depot_y = 500, 500
    plt.scatter(depot_x, depot_y, s=250, c='red', marker='*', label='Depot')

    for (start, end) in road_lines:
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k--', linewidth=1)

    plt.title("Simulated City Delivery Network")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()


def convert_to_json(delivery_points, filename='additional_data.json'):
    base_x, base_y = 500, 500
    base_lon, base_lat = 129.0750875, 35.17982005
    scale = 0.0002

    # 목적지 좌표 변환
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

    # 주문 정보 생성
    box_sizes = [
        {"width": 30, "length": 40, "height": 30},
        {"width": 50, "length": 60, "height": 50},
        {"width": 30, "length": 50, "height": 40}
    ]

    orders = []
    order_number = 1
    box_number = 1
    for dest in destinations:
        num_boxes = random.randint(1, 2)  # 1~2개
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

    # JSON 구성
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
    print(f"[✅] JSON saved to '{filename}'")

# 실행
if __name__ == "__main__":
    total_delivery_points = int(input("Enter total number of delivery points: "))
    points, mains, roads = generate_city_network(total_delivery_points, seed=None)  # 랜덤한 결과
    plot_city_network(points, roads)
    convert_to_json(points)
