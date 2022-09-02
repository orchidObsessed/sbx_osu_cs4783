# =====
# Author: William "Waddles" Waddell
# Class : OKState - CS 4783
# Assmt : 0
# =====
import numpy as np
import matplotlib.pyplot as plt
# =====
def histo(path: str = "/test.txt") -> tuple[dict, np.array]:
    """
    Read a text file and convert it to a histogram as a Python dictionary, and a `NumPy` array.

    Also displays the histogram via `MatPlotLib` as both normalized and non-normalized bar graphs.

    Parameters
    ----------
    `path` : str, default = test.txt
        Path to the file to be read, including extension
    """
    # Step 1: Read file, then tokenize it
    raw = None
    with open(path, "r") as fo:
        raw = fo.readlines() # Get all text
    raw = "".join(raw).lower() # Convert lists of strings to one long string
    raw = [c for c in list(raw) if c.isalpha()] # Tokenize, sans all non-alpha characters

    # Step 2: Iterate through list, build Python dictionary
    freqs = {}
    for c in raw:
        if c in freqs.keys(): freqs[c] += 1 # Increment existing
        else: freqs[c] = 1 # Create if not found

    # Step 3: Sort it, then build NumPy array
    hist = {}
    for k in sorted(freqs): # Re-build py dict by iterating through unsorted one, sorted by keys
        hist[k] = freqs[k]
    print(">>deleteme<< nparr is out of order if not all letters exist, need to pad")
    nphist = np.array(hist.values())

    # Step 4: Build & display MatPLotLibs, and return histogram representations
    fig, (norm, nonnorm) = plt.subplots(2) # 2 graphs

    # Graph non-normalized as a bar graph, using values as y and keys as x
    nonnorm.bar(range(len(hist)), list(hist.values()), align="center", tick_label=list(hist.keys()))
    nonnorm.set_title("Non-Normalized")

    # Graph normalized, normalizing on-the-fly from values (but still using keys as x)
    hmin = min(list(hist.values())) # Minimum
    r = max(list(hist.values())) - hmin # Range
    norm.bar(range(len(hist)), [(x-hmin)/r for x in list(hist.values())], align="center", tick_label=list(hist.keys()))
    norm.set_title("Normalized")

    fig.tight_layout() # This is just to help the output look nicer
    plt.show() # Show it
    print(f"Python Dictionary: {hist}\nNumPy Array: {nphist}")
    return (hist, nphist)

# =====
if __name__ == "__main__":
    histo()
