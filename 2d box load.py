import random
import visualize

class Vehicle:
    X = 16
    Y = 28
    total_volume = X*Y*100

    def __init__(self):
        self.used = [[False]*self.Y for _ in range(self.X)]
        self.box_list = []
        self.box_num = 0

    def load_box_at(self, position, size):
        x, y = position
        size_x, size_y = size
        for dx in range(size_x):
            for dy in range(size_y):
                self.used[x+dx][y+dy] = True

    def unload_box_at(self, position, size):
        x, y = position
        size_x, size_y = size
        for dx in range(size_x):
            for dy in range(size_y):
                self.used[x+dx][y+dy] = False

    def get_depth(self, show=False):
        depth = [0] * self.X
        for x in range(self.X):
            for y in range(self.Y-1, -1, -1):
                if self.used[x][y]:
                    depth[x] = y+1
                    break

        if show:
            for x in range(self.X):
                print(f'{depth[x]:>4d}', end='')
            print()
        return depth

    def calc_possible_volume(self, version):
        volume = [self.Y]*self.X
        depth = self.get_depth()
        size_x = 3
        size_y = 3
        for x in range(self.X-size_x+1):
            d = max(depth[x:x+size_x])
            for dx in range(size_x):
                volume[x+dx] = min(volume[x+dx], d)
            if version == 2 and volume[x] > self.Y - size_y:volume[x] = self.Y

        return self.total_volume - 100 * sum(volume)

    def calc_empty_volume(self):
        volume = 0
        for x in range(self.X):
            for y in range(self.Y):
                if not self.used[x][y]: volume += 100

        return volume

    def calc_filled_volume(self):
        volume = 0
        for x in range(self.X):
            for y in range(self.Y):
                if self.used[x][y]: volume += 100

        return volume

    def load_box_greedy(self, boxes, version):
        def get_possible_positions(size):
            size_x, size_y = size
            depth = self.get_depth()
            positions = []
            for x in range(self.X-size_x+1):
                y = max(depth[x:x+size_x])
                if y + size_y >= self.Y: continue
                positions.append((x, y))

            return positions

        # 30x40x30, 30x50x40, 50x60x50cm
        # 160*280*180 x y z

        self.box_num = 0
        for info in boxes:
            best_fit_size = None
            best_fit_position = None
            best_fit_possible_volume = -1
            for size in get_possible_orientations(info):
                positions = get_possible_positions(size)
                if len(positions) == 0: continue
                for position in positions:
                    self.load_box_at(position, size)
                    possible_volume = self.calc_possible_volume(version)
                    self.unload_box_at(position, size)
                    if possible_volume > best_fit_possible_volume:
                        best_fit_size = size
                        best_fit_position = position
                        best_fit_possible_volume = possible_volume
            if best_fit_position != None:
                self.load_box_at(best_fit_position, best_fit_size)
                self.box_list.append((best_fit_position, best_fit_size))
                self.box_num += 1
            else: break


def random_boxes(n):
    boxes = []
    for _ in range(n):
        boxes.append(random.randint(0, 1))
    return boxes

def get_possible_orientations(info):
    if info == 0:
        return (5, 6), (6, 5)
    else:
        return (5, 4), (4, 5)

def main():
    def handle(event):
        nonlocal index
        if event.key == 'right':
            index = min(index+1, v.box_num)
        elif event.key == 'left':
            index = max(index-1, 0)
        viewer.update(v.box_list[:index])

    # data = []
    # for i in range(100):
    #     r1, r2 = 0, 0
    #     boxes = random_boxes(30)
    #     v = Vehicle()
    #     v.load_box_greedy(boxes, 1)
    #     r1 = v.calc_filled_volume()/v.total_volume
    #     v = Vehicle()
    #     v.load_box_greedy(boxes, 2)
    #     r2 = v.calc_filled_volume()/v.total_volume
    #     print(r1, r2)
    #     data.append(max(r1, r2))
    # print(min(data), max(data))
    # print(sum(data)/100)
    # visualize.histogram(data)

    boxes = random_boxes(30)
    v = Vehicle()
    v.load_box_greedy(boxes, 1)
    r = v.calc_filled_volume()/v.total_volume
    print(r)
    
    index = v.box_num
    viewer = visualize.box_viewer_2d(handle)
    viewer.update(v.box_list)
    viewer.show()

if __name__ == "__main__":
    main()