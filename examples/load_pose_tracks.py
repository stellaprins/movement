"""1. Load pose tracks
======================

Load data from pose tracking software into a movement dataset.
"""

# %%
# Define the path to an input file
# --------------------------------
# This should be a file containing predicted pose tracks, produced by one of
# the supported pose tracking software tools. Normally, you would define a path
# to a file on your system, e.g., :code:`file_path = "/path/to/my/data.h5"`.
# However, in this guide, we will use an example dataset that comes with the
# ``movement`` package.
# sphinx_gallery_thumbnail_path = "_static/example_thumbnail.png"
from movement import sample_data

file_path = sample_data.fetch_sample_data_path(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)
print(f"File path: {file_path}")

# %%
# Load the file into a dataset
# ----------------------------
# We will use the :py:func:`movement.io.load_poses.from_file` function.
# First, we import the :py:mod:`movement.io.load_poses` module and then
# call the function with the file path, the pose tracking software
# that generated the file, and the frame rate of the video (fps).

from movement.io import load_poses

ds = load_poses.from_file(
    file_path,
    source_software="SLEAP",
    fps=50,
)

# %%
# This function will load data from any supported pose tracking software,
# by calling the appropriate loading function, one of:
#
# - :py:func:`movement.io.load_poses.from_dlc_file`
# - :py:func:`movement.io.load_poses.from_sleap_file`
# - :py:func:`movement.io.load_poses.from_lp_file`
#
# Please refer to these for more information on the supported file formats.

# %%
# Inspect the dataset
# -------------------
# The loaded dataset is a :ref:`movement dataset <target-dataset>`, i.e.
# an ` :py:class:`xarray.Dataset` object with some added
# `movement`-specific functionality. Read more about the dataset structure
# in :ref:`this section <target-dataset>`.
#
# Let's see what the loaded dataset contains. If you are working in
# a jupyter notebook, you can simply type the name of the dataset to
# generate an interactive preview (otherwise, use ``print(ds)``).

ds

# %%
# We can see that the dataset ``ds`` was acquired at 50 fps, and the time axis
# is expressed in seconds. It includes data for three individuals
# (``AEON3B_NTP``, ``AEON3B_TP1`` and ``AEON3B_TP2``), and only one keypoint
# called ``centroid`` was tracked in ``x`` and ``y`` coordinates.
# The dataset contains two data variables: ``position`` (predicted keypoint
# coordinates) and ``confidence``.

# %%
# To learn how to visualise the pose tracks, refer to the
# :ref:`sphx_glr_auto_examples_visualise_pose_tracks.py` guide.
