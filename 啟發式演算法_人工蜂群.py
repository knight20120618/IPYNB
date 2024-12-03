# %%
import numpy as np
def ackley_function(position):
    x, y = position
    z = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e+20
    return z

class ABC:
    def __init__(self, func, dim, n_bees, max_iter, bounds, limit):
        self.func = func
        self.dim = dim
        self.n_bees = n_bees # 總蜜蜂數（僱傭蜂 + 觀察蜂）
        self.max_iter = max_iter
        self.bounds = bounds
        self.limit = limit   # 食物源被棄置的次數上限

        # 初始化食物源位置和對應的適應值
        self.positions = np.random.uniform(bounds[0], bounds[1], (n_bees, dim))
        self.fitness = np.array([1 / (1 + func(pos)) for pos in self.positions]) # 適應值
        self.trial_counters = np.zeros(n_bees) # 每個食物源的失敗次數

    def optimize(self):
        for iteration in range(self.max_iter):
            # 僱傭蜂階段[每個僱傭蜂基於現有食物源生成新解]
            for i in range(self.n_bees):
                new_position = self.generate_new_position(i)
                self.update_position(i, new_position)
            # 觀察蜂階段[根據適應值分配資源]
            probs = self.fitness / np.sum(self.fitness)
            for _ in range(self.n_bees):
                i = self.select_food_source(probs)
                new_position = self.generate_new_position(i)
                self.update_position(i, new_position)
            # 偵查蜂階段[如果某個食物源超過失敗限制，隨機重置]
            for i in range(self.n_bees):
                if self.trial_counters[i] > self.limit:
                    self.positions[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                    self.fitness[i] = 1 / (1 + self.func(self.positions[i]))
                    self.trial_counters[i] = 0

            # 顯示當前最佳解
            best_index = np.argmax(self.fitness)
            best_score = 1 / self.fitness[best_index] - 1
            print(f'迭代: {iteration + 1}/{self.max_iter}, 最佳分數: {best_score}')

        # 輸出最佳結果
        best_index = np.argmax(self.fitness)
        return self.positions[best_index], 1 / self.fitness[best_index] - 1

    def generate_new_position(self, i):
        k = np.random.choice([j for j in range(self.n_bees) if j != i])  # 隨機選擇另一個食物源
        phi = np.random.uniform(-1, 1, self.dim)  # 隨機方向
        new_position = self.positions[i] + phi * (self.positions[i] - self.positions[k])
        return np.clip(new_position, self.bounds[0], self.bounds[1])  # 確保新解在邊界內

    def update_position(self, i, new_position):
        new_fitness = 1 / (1 + self.func(new_position))
        if new_fitness > self.fitness[i]:  # 如果新解更優
            self.positions[i] = new_position
            self.fitness[i] = new_fitness
            self.trial_counters[i] = 0  # 重置失敗計數
        else:
            self.trial_counters[i] += 1  # 增加失敗計數

    def select_food_source(self, probs):
        return np.random.choice(range(self.n_bees), p=probs)

dim = 2          # 維度
n_bees = 30      # 蜜蜂數量
max_iter = 100   # 最大迭代次數
bounds = (-5, 5) # 搜尋範圍
limit = 10       # 食物源被棄置的失敗次數上限

abc = ABC(func=ackley_function, dim=dim, n_bees=n_bees, max_iter=max_iter, bounds=bounds, limit=limit)
best_position, best_score = abc.optimize()

print('最佳化位置:', best_position)
print('最佳化分數', best_score)


