# %%
import numpy as np
import random
import math

def sa(x, first_temp, cooling_rate):
    old = np.zeros(x)
    new = np.zeros(x)
    z = 0 # 計算次數
    temp = first_temp # 初始溫度
    
    while new.sum() != x and temp > 1e-3:
        y = random.randint(0, len(new) - 1)
        new[y] = 1
        
        delta = new.sum() - old.sum()
        if delta > 0 or random.random() < math.exp(delta / temp):
            old = new
            print('{} => {}'.format(old, old.sum()))

        temp *= cooling_rate # 冷卻速率
        z += 1
    
    print('共進行了 {} 次迭代'.format(z))

# 幾個值、初始溫度、冷卻速率
sa(10, 100, 0.99)


