'''
This file is only used to test code, not used in the report
'''

# imports
import numpy as np
import pandas as pd
import toolbox as tb
import matplotlib.pyplot as plt

# Variables
alpha = 0.0001
epochs = 35
hidden_layers = 50
output_layer = 10

# Data 
TRNXX = pd.read_csv(r'dataset/TRN_MNIST/TRNXX.csv', header = None)
TRNY1 = pd.read_csv(r'dataset/TRN_MNIST/TRNY1.csv', header = None)
TRNY3 = pd.read_csv(r'dataset/TRN_MNIST/TRNY3.csv', header = None)
TRN10Y = pd.read_csv(r'dataset/TRN_MNIST/TRN10Y.csv', header = None)

RSVXX = pd.read_csv(r'dataset/RSV_MNIST/RSVXX.csv', header = None)
RSVY1 = pd.read_csv(r'dataset/RSV_MNIST/RSVY1.csv', header = None)
RSVY3 = pd.read_csv(r'dataset/RSV_MNIST/RSVY3.csv', header = None)
RSV10Y = pd.read_csv(r'dataset/RSV_MNIST/RSV10Y.csv', header = None)


col_num = 5000

VALXX = RSVXX.iloc[:, :col_num]
TSTXX = RSVXX.iloc[:, col_num:]

VALY3 = RSVY3.iloc[:, :col_num]
TSTY3 = RSVY3.iloc[:, col_num:]

TSTXX.columns = range(5000)
TSTY3.columns = range(5000)

VAL10Y = RSV10Y.iloc[:, :col_num]
TST10Y = RSV10Y.iloc[:, col_num:]

TST10Y.columns = range(5000)

TRNXX = TRNXX.to_numpy()
TRNY1 = TRNY1.to_numpy()
TRNY3 = TRNY3.to_numpy()
TRN10Y = TRN10Y.to_numpy()

RSVXX = RSVXX.to_numpy()
RSVY1 = RSVY1.to_numpy()
RSVY3 = RSVY3.to_numpy()
RSV10Y = RSV10Y.to_numpy()

TSTXX = TSTXX.to_numpy()
TSTY3 = TSTY3.to_numpy()
TST10Y = TST10Y.to_numpy()

def mlpBP(P, T, hidden_layers, alpha, max_epoch, output_layers=1, VALXX=None, VALY=None):

    LCRV = np.zeros((max_epoch))
    validation_LCRV = np.zeros((max_epoch))
    val = False

    rows_p = np.shape(P)[0]
    # (row, col)
    W1 = np.random.randn(hidden_layers, rows_p) / 6
    W2 = np.random.randn(output_layers, hidden_layers) / 6
    b1 = np.random.randn(hidden_layers, 1) / 6
    b2 = np.random.randn(output_layers, 1) / 6

    w1_epoch_i = []
    w2_epoch_i = [] 
    w3_epoch_i = [] 
    w4_epoch_i = [] 
    w5_epoch_i = [] 
    
    if VALXX is not None and VALY is not None:
        val = True
        VALXX = VALXX.to_numpy()
        VALY = VALY.to_numpy()

    for ep in range(max_epoch):
        ep = ep
        W1_new, W2_new, b1_new, b2_new, AE2 = bptans1e(W1, W2, b1, b2, alpha, P, T)

        # Recirculate values
        W1 = W1_new
        W2 = W2_new
        b1 = b1_new
        b2 = b2_new

        LCRV[ep] = AE2

        w1_epoch_i.append(W1_new[0][1]) 
        w2_epoch_i.append(W1_new[0][2]) 
        w3_epoch_i.append(W1_new[0][7]) 
        w4_epoch_i.append(W1_new[0][4]) 
        w5_epoch_i.append(W2_new[0][1]) 

        # After each epoch of training, use the weights and biases obtained to classify the validation set
        if val == True:
            cols_p = np.shape(VALXX)[1]
            hits = 0
            TE2 = 0

            for i in range(cols_p):
                # avg2, hit = predbp1p(W1, W2, b1, b2, P[i], T[i])
                avg2, hit = predbp1p(W1, W2, b1, b2, VALXX[:,i], VALY[:,i])
                TE2 += avg2
                hits += hit

            val_AE2 = TE2 / cols_p
            validation_LCRV[ep] = val_AE2
        else:
            pass

    if val == True: print("validation_LCRV", validation_LCRV)

    # Plotting 

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    if val == True :

        # Plotting Training and Validation MSE
        axs[0].plot(LCRV, label='Training')
        markerline, stemlines, baseline = axs[0].stem(validation_LCRV, linefmt='red', markerfmt='o', label='Validation LCRV')

        markerline.set_markerfacecolor('none')  # Hollow circles
        markerline.set_markeredgecolor('red')  # Marker edge color matches the base color
        markerline.set_markersize(4)  # Adjust marker size

        axs[0].set_title(f'Training & Validation MSE v. Epochs: Alpa = {alpha}')
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("MSE")
        axs[0].set_xlim(0, max_epoch)
        # axs[0].set_ylim(0, np.max(LCRV))
        axs[0].legend(['Training', 'Validation'])

    else:

        # Plotting the MSE per Epoch (Learning Curve)
        axs[0].plot(LCRV)
        axs[0].set_title(f'MSE v. Epoch Learning Curve: Alpa = {alpha}')
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("MSE")
        axs[0].set_xlim(0, max_epoch)
        axs[0].set_ylim(0, np.max(LCRV))

        axs[1].set_xlabel("Epochs")

    # Plotting w1
    (markers, stemlines, baseline) = axs[1].stem(w1_epoch_i)
    plt.setp(stemlines, linestyle='-', color='cyan', linewidth=0 )
    plt.setp(markers, markersize=5, color='cyan', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w2
    (markers, stemlines, baseline) = axs[1].stem(w2_epoch_i)
    plt.setp(stemlines, linestyle='-', color='olive', linewidth=0 )
    plt.setp(markers, markersize=5, color='olive', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w3
    (markers, stemlines, baseline) = axs[1].stem(w3_epoch_i)
    plt.setp(stemlines, linestyle='-', color='green', linewidth=0 )
    plt.setp(markers, markersize=5, color='green', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w4
    (markers, stemlines, baseline) = axs[1].stem(w4_epoch_i)
    plt.setp(stemlines, linestyle='-', color='red', linewidth=0 )
    plt.setp(markers, markersize=5, color='red', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w5
    (markers, stemlines, baseline) = axs[1].stem(w5_epoch_i)
    plt.setp(stemlines, linestyle='-', color='brown', linewidth=0 )
    plt.setp(markers, markersize=5, color='brown', linestyle='-')
    plt.setp(baseline, visible=False)

    axs[1].set_title("5 Weights v Epoch")
    axs[1].legend(['w1', 'w2', 'w3', 'w4', 'w5'])

    # plt.show()

    return W1, W2, b1, b2, AE2



def bptans1e(W1, W2, b1, b2, alpha, P, T):
    cols_p = np.shape(P)[1]

    Te2 = 0
    for i in range(cols_p):
        tester = P[:,i]
        W1_new, W2_new, b1_new, b2_new, avg2 = bptans1pat(W1, W2, b1, b2, alpha, P[:, i], T[:, i])
        Te2 += avg2
    
        # Recirculate Values
        W1 = W1_new
        W2 = W2_new
        b1 = b1_new
        b2 = b2_new
    
    AE2 = Te2 / cols_p

    return W1_new, W2_new, b1_new, b2_new, AE2



def bptans1pat(W1, W2, b1, b2, alpha, p, t):

    # TODO: Change the hardcoded value to exctract the size of the patterns 
    # pat_size = np.shape(p)[0]
    # p = p.reshape(pat_size, 1)

    # Step 1: Forward Propogation and calculation of avg2

    # Input Pattern to Activation of first layer a1

    n1 = np.dot(W1, p) + b1.flatten()
    a1 = np.tanh(n1)

    # a1 to activation of layer 2 (which is the output)

    n2 = np.dot(W2, a1) + b2.flatten()
    a2 = np.tanh(n2)

    numouts = len(t)
    tminusa2 = t - a2 
    tminusa2_reshaped = tminusa2.reshape(np.shape(tminusa2)[0], 1)
    # check later not important v
    avg2 = np.dot(np.transpose(tminusa2_reshaped), tminusa2_reshaped) / numouts 
    

    # Step 2: Backpropogation of sensitivities
    
    #s2 = -2 * tb.fdtansig(n2) * tminusa2 # -2 * np.dot(tb.fdtansig(n2), tminusa2)
    s2 = -2 * np.dot(tb.fdtansig(n2), tminusa2)

    # Backpropogate (s1 is calculated using s2)
    # def: s2 = tb.fdtansig(n1) * W2' * s2

    # PREV: s1 = tb.fdtansig(n1) * np.dot(s2.T, W2)
    s1 = np.dot(tb.fdtansig(n1), np.dot(W2.T, s2))


    # Step 3: Update Weights and Biases
    a1_reshaped = a1.reshape(np.shape(a1)[0], 1)
    s2_reshaped = s2.reshape(np.shape(s2)[0], 1)

    s1_reshaped = s1.reshape(np.shape(s1)[0], 1)
    p_reshaped = p.reshape(np.shape(p)[0], 1)

    W2_new = W2 - alpha * np.dot(s2_reshaped, a1_reshaped.T) # try no Transpose, try reshaping s2
    b2_new = b2 - alpha * s2_reshaped

    W1_new = W1 - alpha * np.dot(s1_reshaped, p_reshaped.T) # s1.T = (50,1)   p.T = 1,784
    b1_new = b1 - alpha * s1_reshaped

    return W1_new, W2_new, b1_new, b2_new, avg2

def predbp1p(W1, W2, b1, b2, p, t):
        
    n1 = np.dot(W1, p) + b1.flatten()
    a1 = np.tanh(n1)

    # a1 to activation of layer 2 (which is the output)

    n2 = np.dot(W2, a1) + b2.flatten()
    a2 = np.tanh(n2)

    numouts = len(t)
    tminusa2 = t - a2 
    tminusa2_reshaped = tminusa2.reshape(np.shape(tminusa2)[0], 1)
    # check later not important v
    avg2 = np.dot(np.transpose(tminusa2_reshaped), tminusa2_reshaped) / numouts 
    # pat_size = np.shape(p)[0]
    # p = p.reshape(pat_size, 1)

    # Input Pattern to Activation of first layer a1
    # n1 = np.dot(W1, p) + b1
    # a1 = np.tanh(n1)

    # # a1 to activation of layer 2 (which is the output)
    # n2 = np.dot(W2, a1) + b2
    # a2 = np.tanh(n2)
    # a2 = np.squeeze(a2)

    # numouts = len(t)

    # tminusa2 = t - a2 # (10,)
    # tminusa2 = tminusa2.reshape(np.shape(tminusa2)[0], 1)
    # avg2 = np.dot(np.transpose(tminusa2), tminusa2) / numouts

    hit = 0
    # Predict based on size of 
    if np.shape(a2)[0] == 10:

        maxa2 = max(a2)
        newa2 = [float(x-maxa2) + 0.001 for x in a2]
        snewa2 = np.sign(newa2)

        if np.array_equal(t, snewa2):
            hit = 1
    else: 
        
        a_thr = tb.hardlims(a2)
        # print("a_thr", a_thr) # delete
        # print("t", t) # delete
        # print("t.item()", t.item()) # delete
        if t.item() == a_thr: 
            hit = 1

    return avg2, hit

# Training
W1_3_3, W2_3_3, b1_3_3, b2_3_3, AE2_3_3 = mlpBP(TRNXX, TRN10Y, hidden_layers, alpha, epochs, output_layers=10, VALXX=VALXX, VALY=VAL10Y)



# Predicting
cols_p = np.shape(TSTXX)[1]


hits = 0
for i in range(cols_p):
    avg2, hit = predbp1p(W1_3_3, W2_3_3, b1_3_3, b2_3_3, TSTXX[:, i], TST10Y[:,i])
    hits += hit
print("Avg2:", avg2)
print("Correct Classifications:", hits)
print("Incorrect Classification", cols_p - hits)
print("Accuracy:", hits / cols_p)

