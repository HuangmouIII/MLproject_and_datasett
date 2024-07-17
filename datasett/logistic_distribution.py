import numpy as np
from scipy.stats import logistic
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(8, 4))  # 创建一个图形和两个子图
x = np.linspace(-5, 5, 100)
ax[0].plot(x, logistic.pdf(x), linewidth=2)  # 在第一个子图上绘制
ax[1].plot(x, logistic.cdf(x), linewidth=2)  # 在第二个子图上绘制累积分布函数作为例子

plt.show()  # 显示图形
