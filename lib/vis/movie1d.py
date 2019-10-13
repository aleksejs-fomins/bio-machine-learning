import numpy as np
import matplotlib.pyplot as plt

from lib.vis.opencv_lib import cvWriter
from lib.vis.matplotlib_helper import fig2numpy


# Glue multiple bar plots into a movie
def time_bars(xx, outfname, dpi=100, figW=600, figH=600):
    nTrial, nNode = xx.shape
    xMin = np.min(xx)
    xMax = np.max(xx)
    barXAxis = np.arange(nNode)

    print("Writing bars movie to", outfname)
    with cvWriter(outfname, (figW, figH), isColor=True) as movieWriter:
        for iFrame, x in enumerate(xx):
            print("Writing video [" + str(iFrame + 1) + '/' + str(nTrial) + ']\r')#, end="")
            fig = plt.figure(figsize=(figW/dpi, figH/dpi), dpi=dpi)
            plt.xlim(0, nTrial)
            plt.ylim(xMin, xMax)
            plt.bar(barXAxis, x)
            movieWriter.write(fig2numpy(fig))
            plt.close(fig)


# Glue multiple scatter plots into a movie
def time_scatter(xx, yy, outfname, dpi=100, figW=600, figH=600):
    nTrial, nNode = xx.shape
    xMin, xMax = np.min(xx), np.max(xx)
    yMin, yMax = np.min(yy), np.max(yy)
    marginX = 0.05*(xMax - xMin)
    marginY = 0.05*(yMax - yMin)

    print("Writing scatter movie to", outfname)
    with cvWriter(outfname, (figW, figH), isColor=True) as movieWriter:
        for iFrame, (x, y) in enumerate(zip(xx, yy)):
            print("Writing video [" + str(iFrame + 1) + '/' + str(nTrial) + ']\r')#, end="")
            fig = plt.figure(figsize=(figW/dpi, figH/dpi), dpi=dpi)
            plt.xlim(xMin - marginX, xMax + marginX)
            plt.ylim(yMin - marginY, yMax + marginY)
            plt.plot(x, y, '.')
            movieWriter.write(fig2numpy(fig))
            plt.close(fig)