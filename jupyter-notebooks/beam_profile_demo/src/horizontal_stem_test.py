# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:18:26 2026

@author: lah
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.arange(32)
y = -0.1 * np.arange(32)

st = ax.stem(y, x, orientation="horizontal")
