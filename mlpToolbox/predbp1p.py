'''
Function to make a prediction using trained weights and biases
'''

# Imports
import numpy as np
import toolbox as tb

def predbp1p(W1, W2, b1, b2, p, t):

    # Input Pattern to Activation of first layer a1   
    n1 = np.dot(W1, p) + b1.flatten()
    a1 = np.tanh(n1)

    # a1 to activation of layer 2 (which is the output)
    n2 = np.dot(W2, a1) + b2.flatten()
    a2 = np.tanh(n2)

    numouts = len(t)
    tminusa2 = t - a2 
    tminusa2_reshaped = tminusa2.reshape(np.shape(tminusa2)[0], 1)
    avg2 = np.dot(np.transpose(tminusa2_reshaped), tminusa2_reshaped) / numouts 

    hit = 0
    # Predict based on size of a2
    if np.shape(a2)[0] == 10:

        maxa2 = max(a2)
        newa2 = [float(x-maxa2) + 0.001 for x in a2]
        snewa2 = np.sign(newa2)

        if np.array_equal(t, snewa2):
            hit = 1
    else: 
        
        a_thr = tb.hardlims(a2)

        if t.item() == a_thr: 
            hit = 1

    return avg2, hit
