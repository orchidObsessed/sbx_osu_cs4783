# =====
# Author: William "Waddles" Waddell
# Class : OKState - CS 4783
# Assmt : 0
# =====
import numpy as np # Since the specific function needed is known, I could import numpy.linalg.solve, but this is better practice. :)
# =====
def terrible_solve_wrapper(a: np.array, b: np.array) -> np.array:
    """
    Given two systems of equations as NumPy arrays (or at least array-likes), use `numpy.linalg.solve` to find a solution.

    This really didn't have to be a function.

    Parameters
    ----------
    a : np.array
        Left part of matrix represnetation of system of equations (coefficients)
    b : np.array
        Right part of matrix represnetation of system of equations
    """
    return np.linalg.solve(a, b)
# =====
if __name__ == "__main__":
    print(terrible_solve_wrapper(np.array([[3, 2], [4, 2]]), np.array([11.25, 10])))
