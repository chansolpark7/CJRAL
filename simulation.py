#조작방법:
#방향키: x z 축 이동.
#n, m : y축 이동.
#스페이스바: 박스 고정 및 새 박스 생성

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

truck_size = [160, 280, 180]

box_types = [
    [30, 40, 30],
    [30, 50, 40],
    [50, 60, 50],
]

box_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']

fixed_boxes = []

current_box = None
current_color = None
current_pos = None

def is_overlapping(pos1, size1, pos2, size2):
    for i in range(3):
        if pos1[i] + size1[i] <= pos2[i] or pos2[i] + size2[i] <= pos1[i]:
            return False
    return True

def is_valid_position(pos, size):
    for other_pos, other_size, _ in fixed_boxes:
        if is_overlapping(pos, size, other_pos, other_size):
            return False
    return True

def find_lowest_valid_y(x, z, size):
    dy = size[1]
    for y in range(0, truck_size[1] - dy + 1):  # bottom to top
        pos = [x, y, z]
        if is_valid_position(pos, size):
            # ensure it's supported by another box or ground
            if y == 0:
                return y
            for other_pos, other_size, _ in fixed_boxes:
                ox, oy, oz = other_pos
                odx, ody, odz = other_size
                if (
                    x < ox + odx and x + size[0] > ox and
                    z < oz + odz and z + size[2] > oz
                ):
                    expected_y = oy + ody
                    if abs(expected_y - y) <= 1:
                        return y
    return None

def draw_box(ax, origin, size, color='cyan', alpha=0.7):
    x, y, z = origin
    dx, dy, dz = size
    vertices = [
        [x,     y,     z],
        [x+dx,  y,     z],
        [x+dx,  y+dy,  z],
        [x,     y+dy,  z],
        [x,     y,     z+dz],
        [x+dx,  y,     z+dz],
        [x+dx,  y+dy,  z+dz],
        [x,     y+dy,  z+dz],
    ]
    faces = [
        [vertices[i] for i in [0,1,2,3]],
        [vertices[i] for i in [4,5,6,7]],
        [vertices[i] for i in [0,1,5,4]],
        [vertices[i] for i in [2,3,7,6]],
        [vertices[i] for i in [1,2,6,5]],
        [vertices[i] for i in [0,3,7,4]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha))

def spawn_new_box():
    global current_box, current_color, current_pos
    for _ in range(200):
        box = random.choice(box_types)
        color = random.choice(box_colors)
        x = 0
        y = 0
        z = 180 - box[2]  # adjusted so box fits
        if is_valid_position([x, y, z], box):
            current_box = box
            current_color = color
            current_pos = [x, y, z]
            return True
    print("더 이상 쌓을 공간이 없습니다.")
    return False

def update_plot():
    ax.cla()
    draw_box(ax, (0, 0, 0), truck_size, color='gray', alpha=0.1)

    for pos, size, color in fixed_boxes:
        draw_box(ax, pos, size, color=color, alpha=0.8)

    if current_pos:
        draw_box(ax, current_pos, current_box, color=current_color, alpha=0.9)

    ax.set_xlim([0, truck_size[0]])
    ax.set_ylim([0, truck_size[1]])
    ax.set_zlim([0, truck_size[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Arrow keys: X/Z | N/M: Y | Space to fix and spawn new')
    plt.draw()

def on_key(event):
    global current_pos

    key = event.key
    dx, dy, dz = 0, 0, 0

    if key == 'left':
        dx = -10
    elif key == 'right':
        dx = 10
    elif key == 'up':
        dz = 10
    elif key == 'down':
        dz = -10
    elif key == 'n':
        dy = -10
    elif key == 'm':
        dy = 10

    if key in ['left', 'right', 'up', 'down', 'n', 'm']:
        new_x = max(0, min(current_pos[0] + dx, truck_size[0] - current_box[0]))
        new_y = max(0, min(current_pos[1] + dy, truck_size[1] - current_box[1]))
        new_z = max(0, min(current_pos[2] + dz, truck_size[2] - current_box[2]))
        if is_valid_position([new_x, new_y, new_z], current_box):
            current_pos = [new_x, new_y, new_z]
            update_plot()

    elif key == ' ':
        fixed_boxes.append((current_pos.copy(), current_box.copy(), current_color))
        if spawn_new_box():
            update_plot()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
fig.canvas.mpl_connect('key_press_event', on_key)

spawn_new_box()
update_plot()
plt.show()
