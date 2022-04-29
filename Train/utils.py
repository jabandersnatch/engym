import matplotlib.pyplot as plt
import numpy as np

def plotLearning(scores, filename, x=None, window=10):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    
    fig, ax = plt.subplots()
    ax.plot(x, running_avg)
    ax.set(xlabel='episode', ylabel='score', title='Running Average')
    ax.grid()
    fig.savefig(filename)
    plt.show(block = False)
    plt.pause(0.001)
    plt.close(fig)
