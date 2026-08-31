# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('../fig/usn-logo-purple.png')

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_axis_off()
ax.set_aspect('equal')  # Forces the correct proportions
