import numpy as np

# 创建一个 50000*512 的随机数组
array = np.random.rand(5, 3)

print(array)
# 计算每列的平均值
mean_array = np.mean(array, axis=0)

print(mean_array)  # 输出应为 (512,)