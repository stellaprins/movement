"""Plotting utilities."""

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Default parameters for scatter plots
scatter_kwargs = {
    "cmap": "viridis",
    "s": 20,
    "edgecolors": None,
    "linewidths": 0,
}

# Default parameters for quiver plots
quiver_kwargs = {
    "angles": "xy",
    "scale": 1,
    "scale_units": "y",
    "cmap": "viridis",
    "width": 0.0125,
}


def _cmap_alpha(cmap: str):
    """Create a colormap with a linear alpha gradient."""
    cmap_ = plt.get_cmap(cmap)  # get the colormap object
    newcolors = cmap_(np.linspace(0, 1, 256))
    alpha_gradient = np.linspace(0, 1, 256)
    newcolors[:, -1] = alpha_gradient
    return ListedColormap(newcolors)


def _crop_axis_lims(ax: plt.Axes, crop: dict[str, tuple[int, int]]):
    """Crop the axis limits of a plot."""
    if "x" in crop:
        ax.set_xlim(*crop["x"])
    if "y" in crop:
        # reverse the y-axis to match the video frame
        crop["y"] = crop["y"][::-1]
        ax.set_ylim(*crop["y"])


def plot_frame_with_keypoints(
    t: int,
    frame: np.ndarray,
    data: xr.DataArray | None = None,
    crop: dict[str, tuple[int, int]] | None = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a frame with keypoints overlaid.

    Parameters
    ----------
    t : int
        The time index of the frame to plot.
    frame : np.ndarray
        The video frame of shape (height, width, channels) to plot.
        Typically this is the frame corresponding to time point ``t``
        in the data.
    data : xr.DataArray
        The input data containing keypoint positions, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
        If None (default), only the video frame is shown.
    crop : dict, optional
        A dictionary containing the x and y limits to crop the plot.
        For example, ``crop={"x": (0, 100), "y": (0, 100)}`` will
        only show the region between x=0 and x=100, and y=0 and y=100.
        Defaults to None (no cropping).
    ax : plt.Axes, optional
        The matplotlib axes to plot on. If not provided, a new figure
        and axes will be created.
    **kwargs
        Additional keyword arguments to pass to
        :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    plt.Axes
        The matplotlib axes containing the plot.

    """
    if ax is None:
        _, ax = plt.subplots()

    # Apply default scatter plot parameters
    for key, value in scatter_kwargs.items():
        kwargs.setdefault(key, value)

    # Plot the video frame
    ax.imshow(frame, aspect="equal")

    # Plot the keypoints if provided
    if data is not None:
        pos_x = data.isel(time=t, space=0)
        pos_y = data.isel(time=t, space=1)
        if "keypoints" in data.dims:
            # color by keypoints if available
            kwargs.setdefault("c", np.arange(data.sizes["keypoints"]))
        else:
            # otherwise assign the same color to all keypoints
            # (last color in the colormap)
            kwargs.setdefault("color", kwargs["cmap"](1.0))
        ax.scatter(pos_x, pos_y, **kwargs)

    if crop is not None:
        _crop_axis_lims(ax, crop)
    ax.axis("off")
    return ax


def plot_trajectory(
    t: int,
    data: xr.DataArray,
    frame: np.ndarray | None = None,
    tail_length: int | None = None,
    tail_alpha_gradient: bool = True,
    crop: dict[str, tuple[int, int]] | None = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a single keypoint track with a tail of previous positions.

    The track is shown from time point ``t - tail_length`` to time point
    ``t``, and can be optionally overlaid on the video frames at time
    point ``t``.

    Parameters
    ----------
    t : int
        The time index at which to plot the track.
    data : xr.DataArray
        The input data containing position information for the track,
        with ``time`` and ``space`` (in Cartesian coordinates) as
        the only dimensions.
    frame : np.ndarray, optional
        The video frame of shape (height, width, channels) to plot.
        Typically this is the frame corresponding to time point ``t``
        in the data. If None (default), only the trajectory is shown.
    tail_length : int, optional
        The number of previous positions to include in the tail.
        If None (default), all previous positions are shown.
        Set to 0 to only show the current position.
    tail_alpha_gradient : bool, optional
        Whether to apply a linear alpha gradient to the tail, ranging
        from fully opaque at time point ``t`` to fully transparent at
        time point ``t - tail_length``. Defaults to True.
    crop : dict, optional
        A dictionary containing the x and y limits to crop the plot.
        For example, ``crop={"x": (0, 100), "y": (0, 100)}`` will
        only show the region between x=0 and x=100, and y=0 and y=100.
        Defaults to None (no cropping).
    ax : plt.Axes, optional
        The matplotlib axes to plot on. If not provided, a new figure
        and axes will be created.
    **kwargs
        Additional keyword arguments to pass to
        :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    plt.Axes
        The matplotlib axes containing the plot.

    """
    if ax is None:
        _, ax = plt.subplots()

    # Apply default scatter plot parameters
    for key, value in scatter_kwargs.items():
        kwargs.setdefault(key, value)

    # Work out the tail length
    tail_length = tail_length or t
    # Apply alpha gradient to the tail if requested
    if tail_alpha_gradient:
        kwargs["cmap"] = _cmap_alpha(kwargs["cmap"])

    # Plot the video frame t, if video is provided
    if frame is not None:
        ax.imshow(frame, aspect="equal")

    # Plot the track using a scatter plot
    show_data = data.isel(time=slice(t - tail_length, t))
    ax.scatter(
        show_data.sel(space="x"),
        show_data.sel(space="y"),
        c=show_data.time.values,
        **kwargs,
    )

    if crop is not None:
        _crop_axis_lims(ax, crop)
    ax.axis("off")
    return ax


def plot_heading_polar(
    t: int,
    heading: xr.DataArray,
    speed: xr.DataArray,
    max_speed: float | None = None,
    tail_length: int | None = None,
    tail_alpha_gradient: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot the heading angle as an arrow in polar coordinates.

    Parameters
    ----------
    t : int
        The time index at which to plot the heading.
    heading : xr.DataArray
        The input data containing the heading angle, with ``time``
        as the only dimension.
    speed : xr.DataArray
        The input data containing the speed of the object, with
        ``time`` as the only dimension. It is used to scale the
        arrow length.
    max_speed : float, optional
        The maximum speed value to use as the upper limit for
        radius scaling. If None (default), the maximum value
        in ``speed.sel(time=(t - tail_length, t))`` is used.
    tail_length : int, optional
        The number of previous headings to include in the tail.
        If None (default), all previous headings are shown.
        Set to 0 to only show the current heading.
    tail_alpha_gradient : bool, optional
        Whether to apply a linear alpha gradient to the tail, ranging
        from fully opaque at time point ``t`` to fully transparent at
        time point ``t - tail_length``. Defaults to True.
    ax : plt.Axes, optional
        The matplotlib axes to plot on. If not provided, a new figure
        and axes will be created.
    **kwargs
        Additional keyword arguments to pass to
        :func:`matplotlib.pyplot.quiver`.

    Returns
    -------
    plt.Axes
        The matplotlib axes containing the plot.

    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Apply default quiver plot parameters
    for key, value in quiver_kwargs.items():
        kwargs.setdefault(key, value)

    # Work out the tail length
    tail_length = tail_length or t
    # Apply alpha gradient to the tail if requested
    if tail_alpha_gradient:
        kwargs["cmap"] = _cmap_alpha(kwargs["cmap"])

    # Angle (phi) and radius (speed) for the quiver plot
    time_slice = slice(t - tail_length, t)
    phi = heading.isel(time=time_slice)
    radius = speed.isel(time=time_slice)

    # Plot the heading as a quiver plot
    ax.quiver(
        phi.values,  # start angle
        np.zeros_like(phi.values),  # start vectors at the origin
        phi.values,  # end angle = start angle
        radius,  # scale the radius by speed
        np.arange(tail_length),  # color by time
        **kwargs,
    )
    ax.set_theta_direction(-1)
    ax.set_theta_offset(0)

    # Control angle axis
    xticks = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(xticks)
    xticks_in_deg = (
        list(range(0, 181, 90)) + list(range(0, -180, -90))[-1:0:-1]
    )
    ax.set_xticklabels([str(t) + "\N{DEGREE SIGN}" for t in xticks_in_deg])

    # Control radius (speed) axis
    max_speed = max_speed or radius.max().item()
    # Round up to the nearest 100
    ymax = 100 * np.ceil(max_speed / 100)
    ax.set_ylim(0, ymax)
    yticks = np.linspace(0, ymax, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(t)) for t in yticks])
    ax.set_rlabel_position(90)
    ax.text(
        0.45,
        0.5,
        "Speed (px/s)",
        transform=ax.transAxes,
        rotation=90,
        ha="right",
        va="center",
    )

    return ax
