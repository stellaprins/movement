"""Filter boolean arrays
===============================

Filter out single-frame interruptions in a 1D boolean array.
"""

# %%
# Imports
# -------

import numpy as np
from scipy.ndimage import convolve

# %%
# Problem formulation
# -------------------
# Let's imagine we are counting the number of interactions between an animal
# and object. We represent these interactions with a booleanr array where
# ``True`` indicates an interaction and ``False`` indicates no interaction.
# We want to filter out single-frame lapses in interaction, i.e.,
# ``False`` values that are surrounded by ``True`` values on both sides
# should be replaced with ``True``.

# %%
# Let's consider the following example. In the array below, we want
# the ``False`` value at index 4 (5th position) to be replaced with ``True``,
# while all other values remain unchanged.

interactions = np.array(
    [False, False, True, True, False, True, False, False, True, False]
)
print(f"Original interactions: {interactions}")

expected_output = np.array(
    [False, False, True, True, True, True, False, False, True, False]
)
print(f"Expected output      : {expected_output}")

# %%
# Solution
# --------
# We can solve this problem by convolving the array with a kernel that
# detects the pattern of interest. We can then use the resulting convolution
# to index into the original array and replace the values that need to be
# changed. First let's define a function that does this.


def filter_out_brief_interruptions(arr: np.ndarray) -> np.ndarray:
    """Filter out single-frame interruptions in a 1D boolean array.

    If a False value is surrounded by True values on both sides, it is
    replaced with True.

    Parameters
    ----------
    arr : np.ndarray
        The input boolean array (must be 1D)

    Returns
    -------
    np.ndarray
        A new boolean array with the filtered data.

    """
    kernel = np.array([1, 1, 1])  # window size 3
    # ensure all arrays as int for convolution
    arr_int = arr.astype(int)
    kernel = kernel.astype(int)
    # convolve the input array with the kernel
    convolved = convolve(arr_int, kernel, mode="constant", cval=0)
    # find the indices where the convolved array is 2 (aka 2 Trues in window)
    # but the central value of the window in the original array is False
    sandwitched_idxs = np.nonzero((convolved == 2) & (arr_int == 0))[0]
    # replace the False values with True at the detected indices
    new_arr = arr.copy()
    new_arr[sandwitched_idxs] = True
    return new_arr


# %%
# Let's test the function on the example input.

filtered_interactions = filter_out_brief_interruptions(interactions)
print(f"Original interactions: {interactions}")
print(f"Expected output      : {expected_output}")
print(f"Filtered interactions: {filtered_interactions}")

assert np.all(filtered_interactions == expected_output)

# %%
# Find bouts of interactions
# --------------------------
# Identify the number and length of bouts of interactions in the filtered
# array. A bout is a continuous sequence of ``True`` values.

# %%
# Define a function that finds the bouts of interactions in the filtered array.
# First we will pad the array with 1 False value at the beginning and end
# to ensure that the first and last bouts are detected.
# Next, we will find the start and end positions of the bouts, using the
# differences between consecutive values.


def find_bouts(arr: np.ndarray) -> np.ndarray:
    """Find bouts of interactions in a 1D boolean array.

    A bout is a continuous sequence of True values.

    Parameters
    ----------
    arr : np.ndarray
        The input boolean array (must be 1D)

    Returns
    -------
    bout_idxs : np.ndarray
        The start and end indices of each bout as a 2D array.
        Each row corresponds to a bout.
        "start" means the first True value in the bout.
        "end" means the first False value after the bout.

    """
    padded_interactions = np.pad(
        arr, 1, mode="constant", constant_values=False
    )
    diffs = np.diff(padded_interactions.astype(int))
    # Start positions are where the diff is 1 (False to True)
    # We subtract 1 to account for padding
    start_positions = np.nonzero(diffs == 1)[0]
    # End positions are where the diff is -1 (True to False)
    end_positions = np.nonzero(diffs == -1)[0]
    # Create a 2D array where each row is a bout
    # and the columns are the start and end indices
    bout_idxs = np.column_stack((start_positions, end_positions))
    return bout_idxs


# %%
# Let's test the function on the filtered interactions.

bout_starts_ends = find_bouts(filtered_interactions)
bout_lengths = bout_starts_ends[:, 1] - bout_starts_ends[:, 0]
print(f"Filtered interactions: {filtered_interactions}")
print(f"Bout lengths         : {bout_lengths}")

for start, end in bout_starts_ends:
    print(f"Bout start: {start}, end: {end}")
    print(f"Bout length: {end - start}")
    bout_content = filtered_interactions[start:end]
    print(f"Bout content: {bout_content}\n")
    assert np.all(bout_content)  # check that all values are True
