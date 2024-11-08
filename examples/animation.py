"""Compute navigation metrics.
==============================

Analyse a  mouse navigating in an elevated plus maze (EPM).
"""

# %%
# Imports
# -------

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import sleap_io as sio
import xarray as xr
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from movement import filtering as filt
from movement import kinematics as kin
from movement.io import load_poses, save_poses
from movement.utils.plotting import (
    plot_frame_with_keypoints,
    plot_heading_polar,
    plot_trajectory,
)
from movement.utils.vector import cart2pol

# %%
# Set up directories
# ------------------

# Project directories
project_dir = Path.home() / "Data" / "behav-analysis-course" / "mouse-EPM"
source_dir = project_dir / "rawdata"
deriv_dir = project_dir / "derivatives"

# Find videos and (1st frame of each video in the source directory
video_files = list(source_dir.glob("*.mp4"))
video_files.sort()

# Find filtered DLC predictions
dlc_dir = deriv_dir / "software-DLC_predictions"
dlc_files = list(dlc_dir.glob("*filtered.h5"))
dlc_files.sort()

# Create a directory to save results and plots
save_dir = project_dir / "derivatives" / "software-movement"
save_dir.mkdir(parents=True, exist_ok=True)

# Construct a dict mapping from subject ID to another dict
# with video, frame, and DLC prediction file paths
subject_ids = [f.name.split("_")[0] for f in video_files]
subject_data = dict()
for i, sub in enumerate(subject_ids):
    subject_data[sub] = {
        "video": video_files[i].as_posix(),
        "dlc": dlc_files[i].as_posix(),
    }
print("Found the following data:")
print(json.dumps(subject_data, sort_keys=True, indent=4))


# %%
# Load data
# ---------
# First let's define a specific subject and time window

sub = "sub-01"
fps = 30  # frames per second

# %%
# Load the DeepLabCut predictions into a movement dataset

ds = load_poses.from_dlc_file(subject_data[sub]["dlc"], fps=fps)
print(ds)

# %%
# Load the video object

video = sio.load_video(subject_data[sub]["video"], plugin="pyav")
n_frames, height, width, channels = video.shape
print(f"Loaded video with shape: {video.shape}")


# %%
# Filter the data
# ---------------

# Let's apply a confidence threshold of 0.9
position_thresh = filt.filter_by_confidence(
    ds.position, ds.confidence, 0.9, print_report=False
)
# Smooth with a median filter
position_smooth = filt.median_filter(
    position_thresh, window=7, min_periods=2, print_report=False
)
# Next, let's linearly interpolate over gaps smaller than 0.5 sec (15 frames)
position_interp = filt.interpolate_over_time(
    position_smooth, max_gap=0.5, print_report=False
)

# Update the position data with its filtered version
ds_filt = ds.copy()
ds_filt.update({"position": position_interp})

# Save the filtered data to disk
save_path = save_dir / f"{sub}_dlc_predictions_filtered.csv"
if save_path.exists():
    save_path.unlink()
save_poses.to_dlc_file(ds_filt, save_path, split_individuals=False)

# %%
# Compute metrics
# ---------------
# First let's select the subset of data we are interested in.

# Keypoints to keep
use_kpts = [
    "snout",
    "left_ear",
    "right_ear",
    "centre",
    "lateral_left",
    "lateral_right",
    "tailbase",
]

# %%
# Now let's compute the navigation metrics we are interested in.

position = ds_filt.position.sel(keypoints=use_kpts)
position.name = "position"

# Compute centroid position
centroid = position.mean(dim="keypoints", skipna=True)
centroid.name = "centroid_position"

# Compute centroid speed
centroid_speed = kin.compute_speed(centroid)
centroid_speed.name = "centroid_speed"

# Compute forward vector and heading (angle)
head_center = position.sel(keypoints=["left_ear", "right_ear", "snout"]).mean(
    dim="keypoints", skipna=True
)
# Forward vector goes from centroid to head_center
forward_vector = head_center - centroid
forward_vector.name = "forward_vector"
# Heading is the phi angle of the forward vector
heading = cart2pol(forward_vector).sel(space_pol="phi", drop=True)
heading.name = "heading"

# %%
# Combine the navigation metrics into a single dataset


def extract_id(data: xr.DataArray) -> xr.DataArray:
    """Drop the 'individuals' dimension from a DataArray."""
    if "individuals" not in data.dims:
        return data
    else:
        return data.sel(individuals="individual_0", drop=True)


position = extract_id(position)
centroid = extract_id(centroid)
centroid_speed = extract_id(centroid_speed)
heading = extract_id(heading)

metrics = xr.merge(
    [position, centroid, centroid_speed, heading],
)
print(metrics)

# %%
# Set Plotting parameters
# -----------------------
# First let's set some plotting parameters.

sns.set_theme(style="ticks", context="poster")

# Set the font to Arial
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"

# %%
# Let's create an animation

# Define the time window for the extracted frames
time_window = (25, 86)  # in seconds (inclusive)
# Get the maximum speed for scaling the polar plot
max_speed = metrics.centroid_speed.sel(time=slice(*time_window)).max().item()
# Define the xlims of the maze (used only for plotting)
crop = {"x": (250, 1100)}

# Define the parameters
tail_length = 90  # in frames
cmap_name = "viridis"
start_t, end_t = (sec * fps for sec in time_window)  # in frames
transition_secs = [7, 14, 21]  # in seconds
transition_ts = [start_t + sec * fps for sec in transition_secs]  # in frames

# %%

# Initialize the figure
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(left=0.0, right=0.95, top=0.94, bottom=0.05, wspace=0.1)
# Create a GridSpec with width ratios
gs = GridSpec(1, 2, width_ratios=[55, 45])
# Initialize the axes using the GridSpec
ax_frame = fig.add_subplot(gs[0, 0])
ax_polar = fig.add_subplot(gs[0, 1], projection="polar")

fig.subplots_adjust(left=0.0, right=0.95, top=0.94, bottom=0.05, wspace=0.1)


def animate(t):
    ax_frame.clear()
    ax_polar.clear()

    frame = video[t]
    t1, t2, t3 = transition_ts

    if t < t1:
        plot_frame_with_keypoints(
            t, frame=frame, data=None, crop=crop, cmap="Set1", ax=ax_frame
        )
        ax_frame.set_title("Mouse moving in a maze")
        ax_polar.axis("off")
    elif t1 <= t < t2:
        plot_frame_with_keypoints(
            t,
            frame=frame,
            data=metrics.position,
            crop=crop,
            cmap="Set1",
            ax=ax_frame,
        )
        ax_frame.set_title("Tracked body parts")
        ax_polar.axis("off")
    elif t2 <= t < t3:
        plot_trajectory(
            t,
            data=metrics.centroid_position,
            frame=frame,
            tail_length=tail_length,
            crop=crop,
            ax=ax_frame,
        )
        ax_frame.set_title("Centroid trajectory")
        ax_polar.axis("off")
    else:
        plot_trajectory(
            t,
            data=metrics.centroid_position,
            frame=frame,
            tail_length=tail_length,
            crop=crop,
            ax=ax_frame,
        )
        ax_frame.set_title("Centroid trajectory")
        plot_heading_polar(
            t,
            heading=metrics.heading,
            speed=metrics.centroid_speed,
            max_speed=max_speed,
            tail_length=tail_length,
            scale=1.5,
            ax=ax_polar,
        )
        ax_polar_title = "Heading (" + "\N{DEGREE SIGN}" + ")"
        ax_polar.set_title(ax_polar_title, pad=18)


# Do a test plot
animate(2000)
plt.savefig(save_dir / f"{sub}_animation_test.png")


# %%
# Create the animation
dt = 1 / fps * 1000  # in milliseconds
ani = FuncAnimation(
    fig, animate, frames=range(start_t, end_t + 1), interval=dt, repeat=False
)
# Save the animation as an mp4 file
ani.save(save_dir / f"{sub}_animation.mp4", writer="ffmpeg", dpi=300)

# %%
