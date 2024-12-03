# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
df = pd.read_csv('2Deil51.csv', sep=' ')
nums = len(df)

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

path = np.zeros((nums, nums))
for i in range(nums):
    for j in range(nums):
        if i != j:
            path[i][j] = distance(df.iloc[i, 1:], df.iloc[j, 1:])

# 蟻群演算法參數
num_ants = 51    # 螞蟻數量
alpha = 1        # 訊息數權重
beta = 2         # 距離權重
rho = 0.5        # 訊息數蒸發率
Q = 100          # 訊息數強度
iterations = 100 # 迭代次數

pheromone = np.ones((nums, nums)) # 初始化訊息數矩陣

def calculate_total(route):
    total_path = 0
    for i in range(len(route) - 1):
        total_path += path[route[i]][route[i + 1]]
    total_path += path[route[-1]][route[0]]
    return total_path

def update_pheromone(ant_routes, ant_distances): # 更新訊息數矩陣
    global pheromone
    pheromone *= (1 - rho)
    for i, route in enumerate(ant_routes):
        for j in range(len(route) - 1):
            pheromone[route[j]][route[j + 1]] += Q / ant_distances[i] # 增加訊息數
        pheromone[route[-1]][route[0]] += Q / ant_distances[i] # 回到起點

def ant_colony():
    best_route = None
    best_distance = float('inf')
    best_distances = []

    for _ in range(iterations):
        ant_routes = []
        ant_distances = []
        
        for _ in range(num_ants):
            start = random.randint(0, nums - 1)
            route = [start]
            visited = set(route)
            
            for _ in range(nums - 1):
                current = route[-1]
                probabilities = [] # 存放各點選擇機率
                nodes = [] # 存放尚未訪問的點
                for next_node in range(nums):
                    if next_node not in visited:
                        tau = pheromone[current][next_node] ** alpha
                        eta = (1 / path[current][next_node]) ** beta
                        probabilities.append(tau * eta)
                        nodes.append(next_node)

                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum() # 機率正規化
                
                next_node = random.choices(nodes, weights=probabilities)[0]
                route.append(next_node)
                visited.add(next_node)
            
            total_distance = calculate_total(route)
            ant_routes.append(route)
            ant_distances.append(total_distance)
            
            if total_distance < best_distance: # 更新最佳路徑
                best_distance = total_distance
                best_route = route
        
        update_pheromone(ant_routes, ant_distances) # 更新訊息數
        best_distances.append(best_distance)

    return best_route, best_distance, best_distances

best_route, best_distance, best_distances = ant_colony()
print('蟻群演算法最佳路線:', best_route)
print('蟻群演算法最短距離:', best_distance)


plt.plot(best_distances) # 收斂圖
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.show()


