"""Compute navigation metrics.
==============================

Analyse a  mouse navigating in an elevated plus maze (EPM).
"""

# %%
# Imports
# -------

from pathlib import Path

from movement import filtering as filt
from movement import kinematics as kin
from movement import sample_data
from movement.io import save_poses
from movement.utils import vector as vec_utils

# %%
# Load data
# ---------

ds = sample_data.fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
print(ds)

# %%
# Filter the data
# ---------------

# Let's apply a confidence threshold of 0.9
position_thresh = filt.filter_by_confidence(
    ds.position, ds.confidence, 0.9, print_report=False
)
# Smooth with a median filter over a 0.2 second window
window = int(0.2 * ds.fps)
position_smooth = filt.median_filter(
    position_thresh, window, min_periods=2, print_report=False
)

# Next, let's linearly interpolate over gaps smaller than 1 second
max_gap = int(1 * ds.fps)
position_interp = filt.interpolate_over_time(
    position_smooth, max_gap=max_gap, print_report=False
)

# Update the position data with its filtered version
ds_filt = ds.copy()
ds_filt.update({"position": position_interp})

# Save the filtered data to disk
project_dir = Path.home() / "Data" / "behav-analysis-course" / "mouse-EPM"
save_dir = project_dir / "derivatives" / "software-movement"
save_dir.mkdir(parents=True, exist_ok=True)

save_poses.to_dlc_file(
    ds_filt,
    save_dir / "DLC_single-mouse_EPM.predictions.filtered.csv",
    split_individuals=False,
)

# %%
# Compute metrics
# ---------------

centroid = ds_filt.position.mean(dim="keypoints", skipna=True)
centroid_speed = kin.compute_speed(centroid)
head_vector = kin.compute_head_direction_vector(
    ds_filt.position,
    left_keypoint="left_ear",
    right_keypoint="right_ear",
    camera_view="top_down",
)
heading = kin.compute_heading(
    ds_filt.position,
    left_keypoint="left_ear",
    right_keypoint="right_ear",
    reference_vector=(1, 0),
    camera_view="top_down",
    in_radians=False,
)

# %%
# Select subset
# -------------
# Remove unnecessary individuals dimension
# And select a subset of the time range.

ds_plot = ds_filt.squeeze(drop=True).sel(time=slice(30, 60))

# %%
# Centroid position
# -----------------

centroid = ds_plot.position.mean(dim="keypoints", skipna=True)
centroid.name = "Centroid position (px)"
centroid.plot.line(x="time", hue="space", aspect=2, size=4)


# %%
# Speed
# -----

speed = kin.compute_speed(centroid)
speed.name = "Centroid speed (px/s)"
speed.plot.line(x="time", aspect=2, size=4)


# %%
# Head direction
# --------------

head_vector = kin.compute_head_direction_vector(
    ds_filt.position,
    left_keypoint="left_ear",
    right_keypoint="right_ear",
    camera_view="top_down",
)

head_vector_phi = vec_utils.cart2pol(head_vector).sel(
    space_pol="phi", drop=True
)


# %%
