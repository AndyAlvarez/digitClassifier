'''
This file contains different helper functions that are useful
'''


#Imports
import numpy as np
import matplotlib.pyplot as plt

def hardlims(n):

    if n >= 0:
        return 1
    else:
        return -1

def fdtansig(n):
    da_dn = 1 - (np.tanh(n) ** 2)
    FDMNM = np.diag(da_dn)
    return FDMNM

def show1mnist(PC):
    PM = np.reshape(PC, (28, 28), order='F')
    mtx = np.flipud(PM)
    r, c = mtx.shape
    ckb = np.zeros((r, c))
    ckb[:r, :c] = mtx

    cmap = plt.cm.gray
    reversed_cmap = cmap.reversed()

    plt.pcolor(ckb, cmap=reversed_cmap)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.colorbar()
    plt.show()
