import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_it(state=None, action=None, sigma_hat=None, uncertainty=None):
    state_num = state.shape[0]
    state_dim = state.shape[1]
    action_num = action.shape[0]
    action_dim = action.shape[1]
    sigma_hat_num = sigma_hat.shape[0]
    sigma_hat_dim = sigma_hat.shape[1]
    uncertainty_num = uncertainty.shape[0]
    uncertainty_dim = uncertainty.shape[1]

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()

    



