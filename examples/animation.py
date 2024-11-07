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
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from movement import filtering as filt
from movement import kinematics as kin
from movement.io import load_poses, save_poses

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

# Find the first frame of each video (showing the maze)
maze_files = list(source_dir.glob("*.png"))
maze_files.sort()

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
        "maze": maze_files[i].as_posix(),
        "dlc": dlc_files[i].as_posix(),
    }
print("Found the following data:")
print(json.dumps(subject_data, sort_keys=True, indent=4))


# %%
# Load data
# ---------

sub = "sub-01"

# Load the DeepLabCut predictions into a movement dataset
ds = load_poses.from_dlc_file(subject_data[sub]["dlc"], fps=30)
# Load the maze image
maze_image = plt.imread(subject_data[sub]["maze"])

# 900 frames extracted from the video (frames 900 - 1800)
frames_dir = save_dir / f"{sub}_frames"
frames_files = list(frames_dir.glob("*.png"))
frames_files.sort()
# Load the 900 png files as a numpy array
frames = np.stack([plt.imread(f) for f in frames_files])

# %%
# Define the time window for the extracted frames
time_window = (30, 60)  # in seconds (fps=30)

# Define the xlims of the maze
maze_xlims = (250, 1100)

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

# Separately save only the selected time window
time_window_str = f"sec-{time_window[0]}-{time_window[1]}"
save_path = save_dir / f"{sub}_dlc_predictions_filtered_{time_window_str}.csv"
if save_path.exists():
    save_path.unlink()
save_poses.to_dlc_file(
    ds_filt.sel(time=slice(*time_window)), save_path, split_individuals=False
)

# %%
# Compute metrics
# ---------------

position = ds_filt.position.sel(time=slice(*time_window), keypoints=use_kpts)
# Reindex the time dimension to start at 0
position = position.assign_coords(time=position.time - position.time[0])

# Compute centroid (of all keypooints) and head_center
centroid = position.mean(dim="keypoints", skipna=True)

# Compute speed (centroid)
centroid_speed = kin.compute_speed(centroid)
centroid_speed.name = "Speed (px/s)"

# Compute head direction vector and heading (angle)
heading_paramts = {
    "left_keypoint": "left_ear",
    "right_keypoint": "right_ear",
    "camera_view": "top_down",
}
forward_vector = kin.compute_forward_vector(position, **heading_paramts)  # type: ignore
heading = kin.compute_heading(
    position,
    **heading_paramts,  # type: ignore
    reference_vector=(1, 0),
    in_radians=True,
)
heading.name = "Heading"

# Drop the "individual" dimension from all metrics
id_sel = {"individuals": "individual_0"}
centroid = centroid.sel(**id_sel, drop=True)
centroid_speed = centroid_speed.sel(**id_sel, drop=True)
forward_vector = forward_vector.sel(**id_sel, drop=True)
heading = heading.sel(**id_sel, drop=True)
max_speed = centroid_speed.max()

# %%
# Set Plotting parameters
# -----------------------

sns.set_theme(style="ticks", context="talk")


def cmap_alpha(cmap_name: str):
    """Create a colormap with a linear alpha gradient."""
    cmap = plt.get_cmap(cmap_name)
    newcolors = cmap(np.linspace(0, 1, 256))
    alpha_gradient = np.linspace(0, 1, 256) ** 2
    newcolors[:, -1] = alpha_gradient
    return ListedColormap(newcolors)


# scatterplot parameters
scatter_params = {
    "s": 20,
    "edgecolors": None,
    "linewidths": 0,
}


# %%
# Plotting functions


def plot_frame(
    t: int,
    show_keypoints: bool = True,
    color: str = "yellow",
    ax: plt.Axes = None,
):
    """Plot a frame with keypoints overlaid."""
    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(frames[t], aspect="equal")
    pos_x = position.isel(time=t, space=0)
    pos_y = position.isel(time=t, space=1)
    if show_keypoints:
        ax.scatter(pos_x, pos_y, color=color, **scatter_params)
    ax.set_xlim(*maze_xlims)
    ax.axis("off")
    return ax


def plot_trajectory(
    t: int,
    tail_length: int = 30,
    cmap_name: str = "viridis",
    ax: plt.Axes = None,
):
    """Plot a trajectory for 30 frames up to time t."""
    if ax is None:
        _, ax = plt.subplots()

    cmap = cmap_alpha(cmap_name)

    # Plot centroid trajectory overlaid on maze
    tail_length = min(t, tail_length)
    show_data = centroid.isel(time=slice(t - tail_length, t))
    ax.imshow(frames[t], aspect="equal")
    ax.scatter(
        show_data.sel(space="x"),
        show_data.sel(space="y"),
        c=show_data.time.values,
        cmap=cmap,
        **scatter_params,
    )

    # Limit the plot to the maze
    ax.set_xlim(*maze_xlims)
    # Turn off axis
    ax.axis("off")
    return ax


def plot_heading(
    t: int,
    tail_length: int = 30,
    cmap_name: str = "viridis",
    ax: plt.Axes = None,
):
    """Plot the heading angle as an arrow in polar coordinates."""
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    tail_length = min(t, tail_length)
    time_slice = slice(t - tail_length, t)
    phi = heading.isel(time=time_slice)
    radius = centroid_speed.isel(time=time_slice)
    ax.quiver(
        phi.values,
        np.zeros_like(phi.values),  # start vectors at the origin
        phi.values,
        radius,  # scale radius by speed
        phi.time.values,  # color by time
        angles="xy",
        scale=1.5,
        scale_units="y",
        cmap=cmap_alpha(cmap_name),
        width=0.0125,
    )
    ax.set_theta_direction(-1)
    ax.set_theta_offset(0)

    # Control radius (speed) axis
    # round up max speed to the nearest 200
    ax.set_ylim(0, centroid_speed.values.max() + 100)
    ax.set_yticks(np.arange(0, max_speed + 200, 200))
    ax.set_rlabel_position(90)
    ax.text(
        0.47,
        0.45,
        centroid_speed.name,
        transform=ax.transAxes,
        rotation=90,
        ha="right",
        va="top",
    )

    # Control angle axis
    xticks = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(xticks)
    xticks_in_deg = (
        list(range(0, 181, 90)) + list(range(0, -180, -90))[-1:0:-1]
    )
    ax.set_xticklabels([str(t) + "\N{DEGREE SIGN}" for t in xticks_in_deg])
    return ax


# %%
# Put it all together is a single animation

# Define the parameters
tail_length = 90
cmap_name = "viridis"
max_sec = 18
max_t = int(max_sec * ds_filt.fps)

# Initialize the figure
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.05, wspace=0.1)
# Create a GridSpec with width ratios
gs = GridSpec(1, 2, width_ratios=[60, 40])  # 60% and 40% widths
# Initialize the axes using the GridSpec
ax_frame = fig.add_subplot(gs[0, 0])
ax_polar = fig.add_subplot(gs[0, 1], projection="polar")


def animate(t):
    color_t = cmap_alpha(cmap_name)(1.0)
    ax_frame.clear()
    ax_polar.clear()

    if t < 4 * 30:
        plot_frame(t, show_keypoints=False, color=color_t, ax=ax_frame)
        ax_frame.set_title("Mouse moving in a maze")
        ax_polar.axis("off")
    elif 4 * 30 <= t < 8 * 30:
        plot_frame(t, show_keypoints=True, color=color_t, ax=ax_frame)
        ax_frame.set_title("Tracked body parts")
        ax_polar.axis("off")
    elif 8 * 30 <= t < 12 * 30:
        plot_trajectory(
            t, tail_length=tail_length, cmap_name=cmap_name, ax=ax_frame
        )
        ax_frame.set_title("Centroid trajectory")
        ax_polar.axis("off")
    else:
        plot_trajectory(
            t, tail_length=tail_length, cmap_name=cmap_name, ax=ax_frame
        )
        ax_frame.set_title("Centroid trajectory")
        plot_heading(
            t, tail_length=tail_length, cmap_name=cmap_name, ax=ax_polar
        )
        ax_polar.set_title("Heading (" + "\N{DEGREE SIGN}" + ")")


# Create the animation
dt = 1 / ds_filt.fps * 1000  # in milliseconds
ani = FuncAnimation(
    fig, animate, frames=range(max_t), interval=dt, repeat=False
)

# To display the animation in a Jupyter notebook, uncomment the following line:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# To save the animation as an MP4 file, uncomment the following lines:
ani.save(save_dir / f"{sub}_animation.mp4", writer="ffmpeg")

# %%
