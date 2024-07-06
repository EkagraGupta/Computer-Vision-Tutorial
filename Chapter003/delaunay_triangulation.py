import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

x, y = np.array(np.random.standard_normal((2, 100)))
tri = Delaunay(np.c_[x, y]).simplices

plt.figure()
plt.plot(x, y, "*")
plt.axis("off")
plt.show()

plt.figure()
for t in tri:
    t_ext = [t[0], t[1], t[2], t[0]]  # add first point to end
    plt.plot(x[t_ext], y[t_ext], "r")
plt.plot(x, y, "*")
plt.axis("off")
plt.show()
