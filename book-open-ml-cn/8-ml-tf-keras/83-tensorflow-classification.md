# Tensorflow的分类算法

## 2 Tensorflow的逻辑回归(Logistic Regression)

### 2.1 构建逻辑回归模型

```python
from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
```

### 2.2 训练逻辑回归模型

### 2.3 调试逻辑回归模型