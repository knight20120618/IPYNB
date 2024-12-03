# %%
import numpy as np
import random

def ts(x, y, a):
    tabu_list = []
    old = np.zeros(x) # 舊解
    new = np.zeros(x) # 新解
    nums = 0 # 迴圈次數
    num = 0 # 特殊規則次數
    while nums < y:
        z = random.randint(0, len(new)-1)
        new[z] = 1
        if (new.sum() > old.sum()) and all(not np.array_equal(new, i) for i in tabu_list):
            if new.sum() > old.sum():
                old = new
                tabu_list.append(new)
                if len(tabu_list) > 7:
                    tabu_list.pop(0)
                print('第{}次: {}, 分數: {}'.format(nums+1, old, old.sum()))
        else:
            num += 1
            if num == a:
                old = max(tabu_list, key=lambda i: i.sum())
                num = 0  # 重製特殊規則次數
                print('第{}次: {}, 分數: {}'.format(nums+1, old, old.sum()))
            else:
                print('第{}次: {}, 分數: {}'.format(nums+1, old, old.sum()))
        nums += 1

# 幾個值、幾個迴圈次數、幾個特殊規則次數
ts(4, 10, 3)


