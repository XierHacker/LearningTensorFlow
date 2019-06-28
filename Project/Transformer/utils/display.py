import numpy as np 
import matplotlib.pyplot as plt 


def draw_positional_encodings(PE):
    plt.pcolormesh(PE[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

