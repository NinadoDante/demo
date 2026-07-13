import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
x = np.linspace(0, 10, 100)

axes[0,0].plot(x, np.sin(x))
axes[0,0].set_title('线图')
axes[0,1].scatter(x, np.cos(x), alpha=0.6)
axes[0,1].set_title('散点图')
axes[1,0].bar(['A','B','C'], [3,7,5])
axes[1,0].set_title('条形图')
axes[1,1].hist(np.random.randn(500), bins=20)
axes[1,1].set_title('直方图')

plt.tight_layout()
plt.savefig('subplots_demo.png', dpi=100)
plt.show()

