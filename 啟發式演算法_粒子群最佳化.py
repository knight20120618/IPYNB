# %%
import numpy as np
def ackley_function(position):
    x, y = position
    z = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e+20
    return z

class PSO:
    def __init__(self, func, dim, n_particles, max_iter, bounds):
        self.func = func
        self.dim = dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.bounds = bounds

        self.positions = np.random.uniform(bounds[0], bounds[1], (n_particles, dim)) # 初始化粒子位置
        self.velocities = np.random.uniform(-1, 1, (n_particles, dim)) # 初始化粒子速度
        
        # 個體最佳位置與全局最佳位置
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.array([func(pos) for pos in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)

    def optimize(self, w=1.5, c1=4, c2=4):
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                # 更新速度
                inertia = w * self.velocities[i]
                cognitive = c1 * np.random.rand() * (self.personal_best_positions[i] - self.positions[i])
                social = c2 * np.random.rand() * (self.global_best_position - self.positions[i])
                self.velocities[i] = inertia + cognitive + social
                
                # 更新位置
                self.positions[i] += self.velocities[i]
                
                # 確保粒子在邊界內
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
                
                # 評估新位置
                score = self.func(self.positions[i])
                
                # 更新個體最佳
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                
                # 更新全局最佳
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
            
            # 顯示迭代過程
            print(f'迭代: {iteration + 1}/{self.max_iter}, 最佳: {self.global_best_score}')
        return self.global_best_position, self.global_best_score

dim = 2          # 維度
n_particles = 30 # 粒子數
max_iter = 100   # 最大迭代次數
bounds = (-5, 5) # 定義搜尋範圍

pso = PSO(func=ackley_function, dim=dim, n_particles=n_particles, max_iter=max_iter, bounds=bounds)
best_position, best_score = pso.optimize()

print('最佳化之位置:', best_position)
print('最佳化之分數:', best_score)


