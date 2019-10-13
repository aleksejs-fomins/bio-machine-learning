import numpy as np
import cv2


class cvWriter:
    # Define the codec and create VideoWriter object

    def __init__(self, filePathName, frameDim, frate=20.0, codec='XVID', isColor=False):
        # For whatever reason OPENCV needs dimensions in the opposite order
        frameDimT = (frameDim[1], frameDim[0])

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._out = cv2.VideoWriter(filePathName, fourcc, frate, frameDimT, isColor=isColor)

    # Just necessary to use 'with' command
    def __enter__(self):
        return self

    # Destructor for the 'with' command
    # Release everything if job is finished
    def __exit__(self, exc_type, exc_value, traceback):
        self._out.release()
        # cv2.destroyAllWindows()

    # Write matrix frame to file
    def write(self, mat):
        self._out.write(mat.astype(np.uint8))