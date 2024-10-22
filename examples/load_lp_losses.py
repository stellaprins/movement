"""Load LightningPose temporal and PCA errors
=============================================

Load LightningPose errors related to temporal smoothness and pose plausibility
into a `movement` poses dataset.
"""

# %%
# Background
# ----------
# LightningPose can output error metrics that are useful for quality control
# and outiler detection. It can be useful to load these metrics into a
# `movement` poses dataset, so that they can be used for filtering the
# predicted poses tracks.
#
# Specifically, we will deal with two types of error metrics:
# - 'temporal_norm.csv': this error is high when the temporal smoothness of
# motion is violated - i.e. when the position of a keypoint changes abruptly
# from one frame to the next.
# - 'pca_singleview_norm.csv': this error is high when the pose is not
# plausible - i.e. when the reprojection of each keypoint through a
# low-dimensional pose subspace (derived via PCA) is not close to the
# original keypoint position.
#
# Each file contains a matrix of errors, where each row corresponds to a
# frame and each column corresponds to a keypoint. The values in the matrix
# are the errors values per keypoint and frame.

# %%
# Imports
# -------
import numpy as np
import pandas as pd
import xarray as xr

from movement.sample_data import fetch_dataset

# %%
# Load a poses dataset
# --------------------
# We will one of our sample poses datasets, which contains predicted poses from
# LightningPose.

ds = fetch_dataset("LP_mouse-face_AIND.predictions.csv")

print(ds)

# %%
# As we can see this dataset contains 12 keypoints on the face of a mouse,
# tracked for 36.000 frames at 60 fps.

# %%
# Load the temporal and PCA errors
# --------------------------------
# First we will create two pandas DataFrames containing random values,
# to simulate the contents of temporal and PCA losses files.
# If you are using LightningPose, load the actual files instead,
# using the `pd.read_csv` function.

# %%
# First, let's generate a random dataframe for the temporal error,
# with shape (n_frames, n_keypoints).

rng = np.random.default_rng(42)

temporal_error = pd.DataFrame(
    rng.random((ds.sizes["time"], ds.sizes["keypoints"])),
    index=np.arange(ds.sizes["time"]),
    columns=ds.keypoints.values,
)

temporal_error.head()

# %%
# Now let's do likewise for the PCA error.

pca_error = pd.DataFrame(
    rng.random((ds.sizes["time"], ds.sizes["keypoints"])),
    index=np.arange(ds.sizes["time"]),
    columns=ds.keypoints.values,
)

pca_error.head()

# %%
# Now let's convert these DataFrames to xarray DataArrays, with shape
# (time, individuals, keypoints) and add them to the dataset.
# For these data, the expected shape is (36.000, 1, 12).

error_shape = tuple(ds.sizes[d] for d in ["time", "individuals", "keypoints"])

ds["temporal_error"] = xr.DataArray(
    temporal_error.values.reshape(error_shape),
    dims=("time", "individuals", "keypoints"),
)

ds["pca_error"] = xr.DataArray(
    pca_error.values.reshape(error_shape),
    dims=("time", "individuals", "keypoints"),
)

# %%
# Now the dataset contains the temporal and PCA errors.

print("\nDataset:\n", ds)

print("\nTemporal error:\n", ds.temporal_error)

print("\nPCA error:\n", ds.pca_error)
