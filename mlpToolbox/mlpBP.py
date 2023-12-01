'''
Function to train and possibly validare a 2 layer MLP with basic backpropogation (BP) for 
as many specified epochs. It also plots the learning curves and 5 weights
'''

# Imports
import matplotlib.pyplot as plt
import numpy as np
import predbp1p
import bptans1e

def mlpBP(P, T, hidden_layers, alpha, max_epoch, output_layers=1, VALXX=None, VALY=None):

    LCRV = np.zeros((max_epoch))
    validation_LCRV = np.zeros((max_epoch))
    val = False

    rows_p = np.shape(P)[0]
    
    # Initialize random weights and biases between -0.5 and 0.5
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

    for ep in range(max_epoch):
        ep = ep
        W1_new, W2_new, b1_new, b2_new, AE2 = bptans1e.bptans1e(W1, W2, b1, b2, alpha, P, T)

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

                avg2, hit = predbp1p.predbp1p(W1, W2, b1, b2, VALXX[:,i], VALY[:,i])
                TE2 += avg2
                hits += hit

            val_AE2 = TE2 / cols_p
            validation_LCRV[ep] = val_AE2
        else:
            pass

    print("Final Validation MSE:", validation_LCRV[-1])

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

    plt.show()

    return W1, W2, b1, b2, AE2

