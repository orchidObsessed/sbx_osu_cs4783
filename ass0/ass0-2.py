# =====
# Author: William "Waddles" Waddell
# Class : OKState - CS 4783
# Assmt : 0
# =====
import cv2
import numpy as np
# =====
def pixops(path: str = "/input.png") -> None:
    """
    Creates two files, output1.png (euclidian distance) and output2.png (blackout square), from a given file.

    Parameters
    ----------
    path : str, default = input.png
        Path to the input file (including extension)
    """
    # Step 1: Load image, create empty NumPy arrays for image outputs (use sample image for sizehint)
    img = np.array(cv2.imread(path)) # Read the image (as a numpy array)
    new_euclid, new_blackout = np.empty_like(img), np.empty_like(img)

    # Step 2: Create Euclidian distance image
    for xcoord in range(img.shape[0]):
        for ycoord in range(img.shape[1]):
            # RBG values as 'unit' vectors along XY, YZ, and XZ planes (using rgb-255 as 'unit')
            unit_vectors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

            # Array of Euclidian distances, using CV2 builtin distance function
            distances = [cv2.norm(np.array(v) - np.array(img[xcoord][ycoord]), cv2.NORM_L2) for v in unit_vectors]

            # Set pixel in new image equal to the plane with least distance from this pixel
            new_euclid[xcoord][ycoord] = unit_vectors[distances.index(min(distances))] # Set corresponding pixel in new image

    # Step 3: Create blackout square image
    center = [int(x/2) for x in new_blackout.shape[0:2]] # (rounded) center pixel

    for xcoord in range(img.shape[0]):
        for ycoord in range(img.shape[1]):
            # If pixel is within square (inclusivity of these ranges leans towards the low end)
            if xcoord in range(center[0]-25, center[0]+25) and ycoord in range(center[1]-25, center[1]+25):
                new_blackout[xcoord][ycoord] = [0, 0, 0] # Set it to black
            else:
                new_blackout[xcoord][ycoord] = img[xcoord][ycoord] # Leave it alone

    # Step 4: Write both images out
    cv2.imwrite("C:\\Users\\Will\\Desktop\\Project Folder\\sandbox\\cs4783\\ass0\\output1.png", new_euclid)
    cv2.imwrite("C:\\Users\\Will\\Desktop\\Project Folder\\sandbox\\cs4783\\ass0\\output2.png", new_blackout)
    return

# =====
if __name__ == "__main__":
    pixops()
