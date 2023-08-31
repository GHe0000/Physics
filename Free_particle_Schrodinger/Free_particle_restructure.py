import numpy as np
import matplotlib.pyplot as plt
import math

from numba import jit

import time

class schrodinger():
    def __init__(self, a, hbar, m, x_start, x_finish, n, infty):
        
        self.hbar = hbar
        self.a = a
        self.m = m
        self.n = n
        
        self.x_start = x_start
        self.x_finish = x_finish
        self.x = np.linspace(-infty, +infty, n)
        self.dx = x[1] - x[0]

        self.x_cac = 0
        self.t_cac = 0

        #self.infty = infty

    def set_t(t):
        self.t_cac = t

    @jit(nopython=True)
    def inte(self, func):
        y = func(self.x)
        return np.sum(y * self.dx)

    @jit(nopython=True)
    def f(k):
        return ((np.sin(self.a * k)) / k)\
                * np.exp(1j * (k * self.x_cac \
                - (self.hbar * (k ** 2)) * self.t_cac))

    @jit(nopython=True)
    def psi(x, t):
        return abs( 1 / \
               (np.pi * np.sqrt(2*self.a))\
                * self.inte(self.f)) ** 2

    def caculate():
        psi = [0] * len(self.x)
        i = -1
        for x_i in x:
            i = i + 1
            psi[i] = psi(x[i], t)
        return psi




