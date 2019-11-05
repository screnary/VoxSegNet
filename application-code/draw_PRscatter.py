"""Draw Precision Recall scatter"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb

Data = np.array([[0.929201491, 0.907628418],
                 [0.924088956, 0.911424081],
                 [0.930117407, 0.910079952],
                 [0.93267417,  0.921156597],
                 [0.937219876, 0.929767623]])

if __name__ == '__main__':
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(linestyle='--', alpha=1.0)
    ax.scatter(Data[0,0], Data[0,1], s=85, marker='o', c='b')  # 75
    ax.scatter(Data[1,0], Data[1,1], s=85, marker='d', c='g')  # 55
    ax.scatter(Data[2,0], Data[2,1], s=85, marker=',')  # 55
    ax.scatter(Data[3,0], Data[3,1], s=85, marker='^')  # 55
    ax.scatter(Data[4,0], Data[4,1], s=150, marker='*', c='r')  # 75

    plt.ylim(0.905, 0.932)
    plt.xlim(0.923, 0.938)
    pdb.set_trace()
    save_dir = os.path.join('Precition_Recall.png')
    plt.savefig(save_dir, dpi=100)
