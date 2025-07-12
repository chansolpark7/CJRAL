from ortools.sat.python import cp_model
import time
import random

def optimal_fill(sequence, H=16, W=28):
    model = cp_model.CpModel()
    placements = []
    
    # 1) 변수 생성
    for p, (bw, bh) in enumerate(sequence):
        for w, h in {(bw,bh), (bh,bw)}:  # 회전
            if w > W or h > H:
                continue
            for r in range(H - h + 1):
                for c in range(W - w + 1):
                    var = model.NewBoolVar(f"x_p{p}_r{r}_c{c}_{w}x{h}")
                    placements.append((p, r, c, w, h, var))

    # 2) 순서당 최대 하나
    for p in range(len(sequence)):
        model.Add(
            sum(var for (pp, *_, var) in placements if pp == p)
            <= 1
        )

    # 3) 비겹침 제약
    # 각 셀 (i,j)에 대해 덮이는 모든 placement 변수 합 ≤ 1
    for i in range(H):
        for j in range(W):
            covering = []
            for (p, r, c, w, h, var) in placements:
                if r <= i < r+h and c <= j < c+w:
                    covering.append(var)
            if covering:
                model.Add(sum(covering) <= 1)

    # 4) 목적 함수
    model.Maximize(sum(var for *_, var in placements))

    # 5) 솔버 실행
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60  # 예: 60초 타임리밋
    solver.parameters.num_search_workers = 1    # 멀티스레드
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("배치된 블록 수 =", solver.ObjectiveValue())
        # 배치 결과 확인
        grid = [[-1]*W for _ in range(H)]
        for (p,r,c,w,h,var) in placements:
            if solver.Value(var):
                for di in range(h):
                    for dj in range(w):
                        grid[r+di][c+dj] = p
        return grid
    else:
        print("해를 찾지 못했습니다.")
        return None

# 사용 예시
# t = time.time()
# sequence = [(3,4)]*20 + [(5,5)]*10
# grid = optimal_fill(sequence)
# print(time.time() - t)

def random_boxes(n):
    boxes = []
    for _ in range(n):
        boxes.append(random.choice((3, 5)))
    return boxes

def solution1(boxes):
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
                    if dp[index + 1][size][j] > dp[index][i][j] + 1:
                        dp[index + 1][size][j] = dp[index][i][j] + 1
                        prev[index + 1][size][j] = (i, j)
                else:
                    if dp[index + 1][i + size][j] > dp[index][i][j]:
                        dp[index + 1][i + size][j] = dp[index][i][j]
                        prev[index + 1][i + size][j] = (i, j)
                if j + size > 18:
                    if dp[index + 1][i][size] > dp[index][i][j] + 1:
                        dp[index + 1][i][size] = dp[index][i][j] + 1
                        prev[index + 1][i][size] = (i, j)
                else:
                    if dp[index + 1][i][j + size] > dp[index][i][j]:
                        dp[index + 1][i][j + size] = dp[index][i][j]
                        prev[index + 1][i][j + size] = (i, j)

    answer = INF
    for i in range(19):
        for j in range(19):
            if dp[n][i][j] != -1:
                answer = min(answer, dp[n][i][j] + int(i != 0) + int(j != 0))

    # print(answer)
    for i in range(19):
        for j in range(19):
            if dp[n][i][j] + int(i != 0) + int(j != 0) == answer: break
        else: continue
        break

    path = []
    for index in range(n, 0, -1):
        prev_i, prev_j = prev[index][i][j]
        if prev_i == i:
            path.append(2)
        else:
            path.append(1)
        i, j = prev_i, prev_j
    path.reverse()
    # print(path)
    return sum(boxes)/(answer*18)

def solution2(boxes):
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
                    if dp[index + 1][i + size][j] > dp[index][i][j]:
                        dp[index + 1][i + size][j] = dp[index][i][j]
                        prev[index + 1][i + size][j] = (i, j)
                if j + size > 18:
                    pass
                else:
                    if dp[index + 1][i][j + size] > dp[index][i][j]:
                        dp[index + 1][i][j + size] = dp[index][i][j]
                        prev[index + 1][i][j + size] = (i, j)

    answer = INF
    for i in range(19):
        for j in range(19):
            if dp[n][i][j] != -1:
                answer = min(answer, dp[n][i][j] + int(i != 0) + int(j != 0))

    # print(answer)
    for i in range(19):
        for j in range(19):
            if dp[n][i][j] + int(i != 0) + int(j != 0) == answer: break
        else: continue
        break

    path = []
    for index in range(n, 0, -1):
        prev_i, prev_j = prev[index][i][j]
        if prev_i == i:
            path.append(2)
        else:
            path.append(1)
        i, j = prev_i, prev_j
    path.reverse()
    # print(path)
    return sum(boxes)/(answer*18)


n = 240

data1 = []
data2 = []
test = 100
for _ in range(test):
    boxes = random_boxes(n)
    data1.append(solution1(boxes))
    data2.append(solution2(boxes))

print(min(data1), max(data1))
print(sum(data1) / test)
print(min(data2), max(data2))
print(sum(data2) / test)

# print(solution1(boxes))
# print(solution2(boxes))