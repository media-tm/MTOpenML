# Tensorflow的回归算法

## 构建随机散点函数

```python
import numpy as np
import matplotlib.pyplot as plt

n = 1024
x = np.random.normal(0, 1, n)  # 平均值为0，方差为1，生成1024个数
y = np.random.normal(0, 1, n)
t = np.arctan2(x, y)  # for color value，对应cmap

plt.scatter(x, y, s=75, c=t, alpha=0.5)   # s为size，按每个点的坐标绘制，alpha为透明度
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xticks([])
plt.yticks([])
plt.show()
```