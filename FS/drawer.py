import matplotlib.pyplot as plt
import numpy as np


def draw(indices, ps, treshold):
    x = np.array([i for i in range(len(ps))])
    y = np.array(ps)
    ok = np.zeros(y.shape)
    ok[indices] = y[indices]

    plt.bar(x, y, color='b')
    plt.bar(x, ok, color='r')
    plt.axhline(y=treshold, color='g', linestyle='--')
    plt.axhline(y=-treshold, color='g', linestyle='--')

    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.title('Features values')
    plt.grid(True)
    plt.show()