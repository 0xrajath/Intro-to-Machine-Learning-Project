import matplotlib.pyplot as plt
from numpy import *
import math

t = linspace(0, 1, 200)
a = -log(t)
b = exp(-t)
plt.plot(t, a, label = 'log loss')
plt.plot(t, b, label = 'exponential loss')
plt.legend(loc='best')
plt.show()
