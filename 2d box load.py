import random
import time
from collections import defaultdict
import visualize

TIME_LIMIT = 10
DEBUG = True

box_sizes= [
    (5, 5),
    (3, 4)
]

box_volumes = [i[0]*i[1] for i in box_sizes]

class LOGER:
    def __init__(self, interval=0.1, enable=True):
        self.interval = interval
        self.enable = enable
        self.last_log = time.time()
        self.datas = []

    def log(self, data):
        if self.enable and time.time() - self.last_log > self.interval:
            self.datas.append(data)
            self.last_log = time.time()

    def return_data(self):
        return self.datas

class Vehicle:
    X = 16
    Y = 28
    total_volume = X*Y*100

    def __init__(self):
        self.used = [[False]*self.Y for _ in range(self.X)]
        self.depth = [0] * self.X
        self.box_list = []
        self.box_num = 0

    def load_box_at(self, position, size):
        x, y = position
        size_x, size_y = size
        for dx in range(size_x):
            for dy in range(size_y):
                self.used[x+dx][y+dy] = True
                self.depth[x+dx] = y+size_y

    def unload_box_at(self, position, size):
        x, y = position
        size_x, size_y = size
        for dx in range(size_x):
            for dy in range(size_y):
                self.used[x+dx][y+dy] = False
        for dx in range(size_x):
            for depth_y in range(self.Y-1, -1, -1):
                if self.used[x+dx][depth_y]:
                    self.depth[x+dx] = depth_y+1
                    break
            else:
                self.depth[x+dx] = 0

    def calc_possible_volume(self, version=2):
        volume = [self.Y]*self.X
        size_x = 3
        size_y = 3
        for x in range(self.X-size_x+1):
            d = max(self.depth[x:x+size_x])
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

    def get_possible_positions(self, size):
        size_x, size_y = size
        positions = []
        for x in range(self.X-size_x+1):
            y = max(self.depth[x:x+size_x])
            if y + size_y > self.Y: continue
            positions.append((x, y))

        return positions

    def load_box_greedy(self, boxes, version):
        # 30x40x30, 30x50x40, 50x60x50cm
        # 160*280*180 x y z

        self.box_num = 0
        for info in boxes:
            best_fit_size = None
            best_fit_position = None
            best_fit_possible_volume = -1
            for size in get_possible_orientations(info):
                positions = self.get_possible_positions(size)
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

    def load_box_bnb(self, boxes_3d):
        # b1 b2 전처리
        b1b2_boxes = [3 if i == 0 else 5 for i in boxes_3d if i != 2]
        def solution(boxes):
            n = len(boxes)
            INF = 300

            dp = [[[INF]*19 for _ in range(19)] for _ in range(n + 1)]
            prev = [[[None]*19 for _ in range(19)] for _ in range(n + 1)]
            dp[0][0][0] = 0

            for index in range(n):
                size = boxes[index]
                for i in range(19):
                    for j in range(19):
                        if dp[index][i][j] == INF: continue
                        if i + size > 18:
                            if dp[index + 1][j][size] > dp[index][i][j] + 1:
                                dp[index + 1][j][size] = dp[index][i][j] + 1
                                prev[index + 1][j][size] = (i, j)
                        else:
                            if dp[index + 1][i + size][j] > dp[index][i][j] + (i == 0):
                                dp[index + 1][i + size][j] = dp[index][i][j] + (i == 0)
                                prev[index + 1][i + size][j] = (i, j)
                        if j + size > 18:
                            pass
                        else:
                            if dp[index + 1][i][j + size] > dp[index][i][j] + (j == 0):
                                dp[index + 1][i][j + size] = dp[index][i][j] + (j == 0)
                                prev[index + 1][i][j + size] = (i, j)

            answer = min(min(arr) for arr in dp[n])

            best_i, best_j = None, None
            best_score = -1
            for i in range(19):
                for j in range(19):
                    if dp[n][i][j] == answer:
                        score = (18 - i if i <= 15 else 0) + (18 - j if j <= 15 else 0)
                        if score > best_score:
                            best_i, best_j = i, j
                            best_score = score

            path = []
            box_index = answer-1
            i, j = best_i, best_j
            for index in range(n, 0, -1):
                size = boxes[index-1]
                prev_i, prev_j = prev[index][i][j]
                if prev_i == i and prev_j + size == j:
                    path.append(box_index)
                else:
                    if prev_i + size > 18:
                        path.append(box_index)
                        box_index -= 1
                    else:
                        path.append(box_index-1)
                i, j = prev_i, prev_j
            path.reverse()
            return path
        b1b2_path = solution(b1b2_boxes)
        b3_path = [i//3 for i in range(len(boxes_3d) - len(b1b2_boxes))]

        boxes = []
        b1b2_counter = -1
        b1b2_index = 0
        b3_counter = -1
        b3_index = 0
        for index, info in enumerate(boxes_3d):
            if info < 2:
                if b1b2_path[b1b2_index] != b1b2_counter:
                    boxes.append(0)
                b1b2_index += 1
            else:
                if b3_path[b3_index] != b3_counter:
                    boxes.append(1)
                b3_index += 1

        answer_volume = 0
        answer_box_list = None
        self.boxes = boxes
        self.box_num = 0
        start_t = time.time()
        visited = defaultdict(set)
        loading = []
        d = defaultdict(list)
        for index, info in enumerate(boxes): # loading중인 박스 처리 나중에 수정
            loading.append(d[0] + d[1])
            if info == 0:
                if len(d[0]) == 1: d[0].pop(0)
            else:
                if len(d[1]) == 2: d[1].pop(0)
            d[info].append(index)
        print(loading)

        def estimate(index):
            possible_volume = self.calc_possible_volume()
            volume = 0
            for info in self.boxes[index:]:
                if volume + box_volumes[info] <= possible_volume:
                    volume += box_volumes[info]
                else:
                    break
            return volume
        
        def dfs(index, volume):
            nonlocal answer_volume
            nonlocal answer_box_list

            if time.time() - start_t > TIME_LIMIT: return
            log.log(answer_volume/self.total_volume)
            if index == len(self.boxes): return
            if DEBUG and index < 3: print(index)

            estimated_volume = estimate(index)
            if volume + estimated_volume <= answer_volume: return

            info = self.boxes[index]
            box_volume = box_volumes[info]

            best_fit_boxes = None
            best_fit_possible_volume = -1
            for size in get_possible_orientations(info):
                positions = self.get_possible_positions(size)
                for position in positions:
                    self.load_box_at(position, size)
                    self.box_list.append((position, size))
                    possible_volume = self.calc_possible_volume()
                    if best_fit_possible_volume < possible_volume:
                        best_fit_possible_volume = possible_volume
                        best_fit_boxes = [(position, size)]
                    elif best_fit_possible_volume == possible_volume:
                        best_fit_boxes.append((position, size))
                    self.unload_box_at(position, size)
                    self.box_list.pop()
                    if volume + box_volume > answer_volume:
                        answer_volume = volume + box_volume
                        answer_box_list = self.box_list + [(position, size)]
            if best_fit_boxes != None:
                not_shuffling = []
                for position, size in best_fit_boxes:
                    for b_index in loading[index]:
                        b_info = self.boxes[b_index]
                        b_position, b_size = self.box_list[b_index]
                        if info == b_info == 0: continue
                        elif position[0] + size[0] <= b_position[0] or b_position[0] + b_size[0] <= position[0]: continue
                        else: break
                    else:
                        not_shuffling.append((position, size))
                if len(not_shuffling) > 0: best_fit_boxes = not_shuffling
                elif index == 0: print(not_shuffling)
                # best_fit_boxes = not_shuffling

                for position, size in best_fit_boxes:
                    if position in visited[size]: continue
                    self.load_box_at(position, size)
                    self.box_list.append((position, size))
                    dfs(index + 1, volume + box_volume)
                    self.unload_box_at(position, size)
                    self.box_list.pop()
                    visited[size].add(position)
                for position, size in best_fit_boxes:
                    visited[size].discard(position)

        dfs(0, 0)
        self.box_list = answer_box_list
        self.box_num = len(self.box_list)
        for position, size in self.box_list:
            self.load_box_at(position, size)

def random_boxes(n):
    boxes = []
    for _ in range(n):
        boxes.append(random.randint(0, 2))
    return boxes

def get_possible_orientations(info):
    if info == 0:
        return (5, 5),
    else:
        return (3, 4), (4, 3)

def main():
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
    # boxes = [0]*20 + [1]*10
    # random.shuffle(boxes)
    # boxes = [1]*60
    print(boxes)

    v = Vehicle()
    t = time.time()
    # v.load_box_greedy(boxes, 2)
    v.load_box_bnb(boxes)
    print(time.time() - t)
    r = v.calc_filled_volume()/v.total_volume
    print(r)
    # print(v.box_list)
    # print(v.box_num)

    index = v.box_num
    viewer = visualize.box_viewer_2d(v.box_list)
    viewer.show()

    visualize.graph(data=log.return_data())

if __name__ == "__main__":
    log = LOGER()
    # for _ in range(10):
    main()