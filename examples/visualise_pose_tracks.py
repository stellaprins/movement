"""Visualise pose tracks
============================

Visualise pose tracks using xarray and matplotlib.

Here we rely on ``xarray``'s built-in
`plotting <https://docs.xarray.dev/en/stable/user-guide/plotting.html>`_
methods as well as some custom ``matplotlib`` plotting functions to visualise
pose tracks.
"""

# %%
# Load a dataset
# --------------------------
# Here we load an example dataset that comes with the ``movement``.
# If you wish to load your own data from a file, refer to the
# :ref:`sphx_glr_auto_examples_load_pose_tracks.py` guide.
# sphinx_gallery_thumbnail_number = 2

from movement import sample_data

ds = sample_data.fetch_sample_data(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)

ds

# %%
# We can see that the dataset ``ds`` was acquired at 50 fps, and the time axis
# is expressed in seconds. It includes data for three individuals
# (``AEON3B_NTP``, ``AEON3B_TP1`` and ``AEON3B_TP2``), and only one keypoint
# called ``centroid`` was tracked in ``x`` and ``y`` coordinates.
# The dataset contains two data variables: ``position`` (predicted keypoint
# coordinates) and ``confidence``.


# %%
# Visualise keypoint positions over time
# --------------------------------------
# ``xarray``'s built-in
# `plotting <https://docs.xarray.dev/en/stable/user-guide/plotting.html>`_
# methods make it very easy for us.

position = ds.position.squeeze()  # remove the "keypoints" dimension (size 1)
position.plot.line(
    x="time", row="individuals", hue="space", aspect=2, size=2.5
)

# %%
# Plot trajectories
# -----------------
# Here we will use ``matplotlib`` to plot the trajectories of the mice in the
# XY plane, colouring them by individual.

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 1)
for mouse_name, col in zip(position.individuals.values, ["r", "g", "b"]):
    ax.plot(
        position.sel(individuals=mouse_name, space="x"),
        position.sel(individuals=mouse_name, space="y"),
        c=col,
        label=mouse_name,
    )
    ax.invert_yaxis()
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.axis("equal")
    ax.legend()

# %%
# We can see that the trajectories of the three mice are close to a circular
# arc. Notice that the x and y axes are set to equal scales, and that the
# origin of the coordinate system is at the top left of the image. This
# follows the convention for SLEAP and most image processing tools.

# %%
# We can also color the data points based on their timestamps:
fig, axes = plt.subplots(3, 1, sharey=True)
for mouse_name, ax in zip(position.individuals.values, axes):
    sc = ax.scatter(
        position.sel(individuals=mouse_name, space="x"),
        position.sel(individuals=mouse_name, space="y"),
        s=2,
        c=position.time,
        cmap="viridis",
    )
    ax.invert_yaxis()
    ax.set_title(mouse_name)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.axis("equal")
    fig.colorbar(sc, ax=ax, label="time (s)")
fig.tight_layout()

# %%
# These plots show that for this snippet of the data,
# two of the mice (``AEON3B_NTP`` and ``AEON3B_TP1``)
# moved around the circle in clockwise direction, and
# the third mouse (``AEON3B_TP2``) followed an anti-clockwise direction.
