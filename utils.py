import math
import numpy as np
import matplotlib.pyplot as plt

def plot_std(x, std, c='C0', alpha=.3):
    log10e = math.log10(math.exp(1))
    P = x.keys()
    rerr = log10e *  std / x
    plt.fill_between(P, 10 ** (np.log10(x) - rerr), 10 ** (np.log10(x) + rerr), color=c, alpha=alpha)