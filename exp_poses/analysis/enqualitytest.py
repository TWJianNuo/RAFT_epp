import numpy as np

a = np.random.random(20)
b = np.random.random(20)
c = np.random.random(20)
d = np.random.random(20)
val1 = (a / c + b / d)  / 2
val2 = (a + b) / (c + d)
print(np.sum((val1 - val2) >= 0))