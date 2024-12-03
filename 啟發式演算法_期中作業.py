# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('./3Deil101.csv', sep=' ')

def distance(a, b): # 計算兩點距離
    return np.sqrt(np.sum((a - b) ** 2))

nums = len(df) # 計算座標數量
path = np.zeros((nums, nums)) # 初始化距離矩陣

for i in range(nums): # 使用雙重迴圈計算距離
    for j in range(nums):
        if i != j:
            path[i][j] = distance(df.iloc[i, 1:], df.iloc[j, 1:])

def calculate_total(route): # 計算路線總距離
    total_path = 0 # 初始化總距離
    for i in range(len(route) - 1):
        total_path += path[route[i]][route[i + 1]]
    total_path += path[route[-1]][route[0]]
    return total_path

def hill_climbing(): # 爬山演算法
    current_route = list(range(nums))
    random.shuffle(current_route)
    
    current_distance = calculate_total(current_route) # 計算初始路線總距離
    hc = [current_distance] # 追蹤距離變化LIST

    while True:
        neighbors = [] # 儲存鄰居解
        for i in range(nums):
            for j in range(i + 1, nums):
                neighbor = current_route[:] # 複製當前路線
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i] # 交換位置 i 和 j
                neighbors.append(neighbor)

        best_neighbor = None # 初始化最佳鄰居
        best_distance = current_distance # 初始化最佳距離

        for neighbor in neighbors:
            dist = calculate_total(neighbor)
            if dist < best_distance:
                best_distance = dist
                best_neighbor = neighbor
        
        if best_neighbor is None:
            break
        else:
            current_route = best_neighbor
            current_distance = best_distance
            hc.append(current_distance) # 紀錄最短距離

    return current_route, current_distance, hc # 最佳路線、最短距離、距離變化過程

current_route_hc, current_distance_hc, hc = hill_climbing()
print('爬山演算法最佳路線:', current_route_hc)
print('爬山演算法最短距離:', current_distance_hc)


fig_hc = plt.figure() # 3D 拓樸圖
ax_hc = fig_hc.add_subplot(111, projection='3d')
x_hc = df.iloc[:, 1].values
y_hc = df.iloc[:, 2].values
z_hc = df.iloc[:, 3].values

ax_hc.scatter(x_hc, y_hc, z_hc, c='r', marker='*')
for i in range(len(current_route_hc)):
    start = current_route_hc[i]
    end = current_route_hc[(i + 1) % len(current_route_hc)]
    ax_hc.plot([x_hc[start], x_hc[end]], [y_hc[start], y_hc[end]], [z_hc[start], z_hc[end]], c='g')

# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('./3Deil101.csv', sep=' ')

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

nums = len(df)
path = np.zeros((nums, nums))

for i in range(nums):
    for j in range(nums):
        if i != j:
            path[i][j] = distance(df.iloc[i, 1:], df.iloc[j, 1:])

def calculate_total(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += path[route[i]][route[i + 1]]
    total_distance += path[route[-1]][route[0]]
    return total_distance

def simulated_annealing(initial_temp=100, cooling_rate=0.99):
    current_route = list(range(nums))
    random.shuffle(current_route)
    current_distance = calculate_total(current_route)
    
    best_route = current_route[:]
    best_distance = current_distance
    sa = [best_distance]

    temp = initial_temp

    while temp > 1:
        i, j = random.sample(range(nums), 2)
        neighbor = current_route[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_distance = calculate_total(neighbor)

        distance_diff = neighbor_distance - current_distance

        if distance_diff < 0 or random.uniform(0, 1) < np.exp(-distance_diff / temp):
            current_route = neighbor
            current_distance = neighbor_distance

            if current_distance < best_distance:
                best_distance = current_distance
                best_route = current_route[:]
                sa.append(best_distance)

        temp *= cooling_rate

    return best_route, best_distance, sa

best_route_sa, best_distance_sa, sa = simulated_annealing()
print('模擬退火演算法最佳路線:', best_route_sa)
print('模擬退火演算法最短距離:', best_distance_sa)


fig_sa = plt.figure() # 3D 拓樸圖
ax_sa = fig_sa.add_subplot(111, projection='3d')
x_sa = df.iloc[:, 1].values
y_sa = df.iloc[:, 2].values
z_sa = df.iloc[:, 3].values

ax_sa.scatter(x_sa, y_sa, z_sa, c='r', marker='*')
for i in range(len(best_route_sa)):
    start = best_route_sa[i]
    end = best_route_sa[(i + 1) % len(best_route_sa)]
    ax_sa.plot([x_sa[start], x_sa[end]], [y_sa[start], y_sa[end]], [z_sa[start], z_sa[end]], c='g')

# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('./3Deil101.csv', sep=' ')

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

nums = len(df)
path = np.zeros((nums, nums))

for i in range(nums):
    for j in range(nums):
        if i != j:
            path[i][j] = distance(df.iloc[i, 1:], df.iloc[j, 1:])

def calculate_total(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += path[route[i]][route[i + 1]]
    total_distance += path[route[-1]][route[0]]
    return total_distance

def tabu_search(max_iterations=100, tabu_tenure=7):
    current_route = list(range(nums))
    random.shuffle(current_route)

    current_distance = calculate_total(current_route)
    best_route = current_route[:]
    best_distance = current_distance
    ts = [best_distance]
    
    tabu_list = []
    
    for _ in range(max_iterations):
        neighbors = []
        for i in range(nums):
            for j in range(i + 1, nums):
                if (i, j) not in tabu_list:
                    neighbor = current_route[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append(neighbor)
        
        best_neighbor = None
        best_neighbor_distance = float('inf')
        
        for neighbor in neighbors:
            distance = calculate_total(neighbor)
            if distance < best_neighbor_distance:
                best_neighbor_distance = distance
                best_neighbor = neighbor
        
        if best_neighbor is not None:
            current_route = best_neighbor
            current_distance = best_neighbor_distance
            
            if current_distance < best_distance:
                best_distance = current_distance
                best_route = current_route
                ts.append(best_distance)
            
            tabu_list.append((current_route[i], current_route[j]))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

    return best_route, best_distance, ts

best_route_ts, best_distance_ts, ts = tabu_search()
print('禁忌搜尋法最佳路線:', best_route_ts)
print('禁忌搜尋法最短距離:', best_distance_ts)


fig_ts = plt.figure() # 3D 拓樸圖
ax_ts = fig_ts.add_subplot(111, projection='3d')
x_ts = df.iloc[:, 1].values
y_ts = df.iloc[:, 2].values
z_ts = df.iloc[:, 3].values

ax_ts.scatter(x_ts, y_ts, z_ts, color='r', marker='*')
for i in range(nums):
    start = (x_ts[i], y_ts[i], z_ts[i])
    end = (x_ts[(i + 1) % nums], y_ts[(i + 1) % nums], z_ts[(i + 1) % nums])
    ax_ts.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='g')

# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('./3Deil101.csv', sep=' ')

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

nums = len(df)
path = np.zeros((nums, nums))

for i in range(nums):
    for j in range(nums):
        if i != j:
            path[i][j] = distance(df.iloc[i, 1:], df.iloc[j, 1:])

def calculate_total(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += path[route[i]][route[i + 1]]
    total_distance += path[route[-1]][route[0]]
    return total_distance

def create_population(pop_size):
    population = []
    for _ in range(pop_size):
        route = list(range(nums))
        random.shuffle(route)
        population.append(route)
    return population

def select_parents(population):
    sorted_population = sorted(population, key=calculate_total)
    return sorted_population[:2]

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    pointer = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[pointer] in child:
                pointer += 1
            child[i] = parent2[pointer]
    return child

def mutate(route):
    if random.random() < 0.1:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]

def genetic_algorithm(pop_size=200, generations=100):
    population = create_population(pop_size)
    ga = []

    for _ in range(generations):
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population

        best_distance = calculate_total(min(population, key=calculate_total))
        ga.append(best_distance)

    best_route = min(population, key=calculate_total)
    best_distance = calculate_total(best_route)
    return best_route, best_distance, ga

best_route_ga, best_distance_ga, ga = genetic_algorithm()
print('基因演算法最佳路線:', best_route_ga)
print('基因演算法最短距離:', best_distance_ga)


fig_ga = plt.figure() # 3D 拓樸圖
ax_ga = fig_ga.add_subplot(111, projection='3d')
x_ga = df.iloc[:, 1].values
y_ga = df.iloc[:, 2].values
z_ga = df.iloc[:, 3].values

ax_ga.scatter(x_ga, y_ga, z_ga, c='r', marker='*')
for i in range(nums):
    start = best_route_ga[i]
    end = best_route_ga[(i + 1) % nums]
    ax_ga.plot([x_ga[start], x_ga[end]], [y_ga[start], y_ga[end]], [z_ga[start], z_ga[end]], c='g')

# %%
plt.plot(hc, marker='*', c='r', label='HC')
plt.plot(sa, marker='*', c='y', label='SA')
plt.plot(ts, marker='*', c='g', label='TS')
plt.plot(ga, marker='*', c='b', label='GA')

plt.title('ConvergencePlot')
plt.xlabel('Iteration')
plt.ylabel('TotalDistance')
plt.legend()
plt.show()


