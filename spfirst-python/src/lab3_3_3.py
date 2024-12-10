# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:16:15 2024

@author: lah
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

b = 1/3*np.ones(3)

x = np.append(np.ones(10), np.zeros(5))

n_start = 0
n_end = len(x)

n = np.arange(n_start, n_end)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.stem(n, x[n])
ax1.set_xlabel("n")
ax1.set_ylabel("x[n]")

y = signal.convolve(x, b)

ax2 = fig.add_subplot(2, 1, 2)
ax2.stem(n, y[n])
ax2.set_xlabel("n")
ax2.set_ylabel("y[n] = x[n]*h[n]")
