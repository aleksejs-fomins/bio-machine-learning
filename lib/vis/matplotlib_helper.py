import numpy as np
import matplotlib.pyplot as plt


def fig2numpy(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
