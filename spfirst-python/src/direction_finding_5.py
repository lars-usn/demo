# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:29:32 2024

@author: lah
"""

# Imports and defs
import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, exp, sqrt
import zplot


def find_phase(xv, xr, frequency=400):
    "Find phase for microphone at position xr"
    yr = 100
    c = 340

    r = np.sqrt((xv - xr)**2 + yr**2)
    t = r/c

    phi = -2*pi*frequency*t

    return phi


# Dimensions
c = 340
f = 400
A = 1

d = 0.40
xv = np.arange(-400, 400, 1)
yr = 100

phi_1 = find_phase(xv, -1/2*d, frequency=f)
phi_2 = find_phase(xv, +1/2*d, frequency=f)

d_phi = phi_2 - phi_1

plt.plot(xv, d_phi)
plt.xlabel("Vehicle position $x_v$ [m]")
plt.ylabel("Phase difference  [rad]")
plt.grid("on")

# Phasors
X1 = A*np.exp(1j*phi_1)
X2 = A*np.exp(1j*phi_2)

theta = np.array([])
theta = np.arctan(xv/yr)
d_phi_f = 2*pi*f * d * np.sin(theta)/c
d_phi_x = 2*pi*f * d /c


plt.plot(xv, d_phi_f, '--')
