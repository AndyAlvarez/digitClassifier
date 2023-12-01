'''
Function to train one epoch of a 2 layer MLP with basic backpropogation (BP)
'''

# Imports
import numpy as np
import bptans1pat

def bptans1e(W1, W2, b1, b2, alpha, P, T):

    cols_p = np.shape(P)[1]
    Te2 = 0
    
    for i in range(cols_p):

        W1_new, W2_new, b1_new, b2_new, avg2 = bptans1pat.bptans1pat(W1, W2, b1, b2, alpha, P[:, i], T[:, i])
        Te2 += avg2
    
        # Recirculate Values
        W1 = W1_new
        W2 = W2_new
        b1 = b1_new
        b2 = b2_new
    
    AE2 = Te2 / cols_p

    return W1_new, W2_new, b1_new, b2_new, AE2