"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from movement.io import PoseTracks

# get logger
logger = logging.getLogger(__name__)


def napari_get_reader(path: Union[str, list[str]]) -> Optional[Callable]:
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if Path(path).suffix not in [".h5", ".hdf5", ".slp", ".csv"]:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(
    path: Union[str, list[str]]
) -> list[tuple[np.ndarray[Any, Any], dict[str, Any], str]]:
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    layers = []

    for path in paths:
        pose_tracks = PoseTracks.from_sleap_file(path, fps=60)
        data, props = pose_tracks_to_napari_tracks(pose_tracks)
        logger.info(f"Converted pose tracks from {path} into napari tracks.")
        logger.debug(f"Tracks data shape: {data.shape}")

        # optional kwargs for the corresponding viewer.add_* method
        add_kwargs = {
            "name": Path(path).name,
            "properties": props,
            "visible": True,
            "tail_width": 5,
            "tail_length": 100,
            "colormap": "turbo",
            "color_by": "individual",
        }

        layers.append((data, add_kwargs, "tracks"))

    return layers


def pose_tracks_to_napari_tracks(
    ds: PoseTracks,
) -> tuple[np.ndarray, dict]:
    """Converts a PoseTracks object to a numpy array of formatted
    for the napari tracks layer."""

    n_frames = ds.dims["time"]
    n_individuals = ds.dims["individuals"]
    n_keypoints = ds.dims["keypoints"]
    n_tracks = n_individuals * n_keypoints

    # assign unique integer ids to individuals and keypoints
    ds.coords["individual_ids"] = ("individuals", range(n_individuals))
    ds.coords["keypoint_ids"] = ("keypoints", range(n_keypoints))

    # Convert 4D to 2D array by stacking
    ds = ds.stack(tracks=("individuals", "keypoints", "time"))
    # track ids are unique integers (individual_id * n_keypoints + keypoint_id)
    individual_ids = ds.coords["individual_ids"].values
    keypoint_ids = ds.coords["keypoint_ids"].values
    track_ids = individual_ids * n_keypoints + keypoint_ids

    yx_columns = np.fliplr(ds.pose_tracks.values.T)
    time_column = np.tile(np.arange(n_frames), n_tracks)
    napari_tracks = np.hstack(
        (track_ids.reshape(-1, 1), time_column.reshape(-1, 1), yx_columns)
    )

    scores = ds.confidence_scores.values.flatten()
    properties = {
        "confidence_scores": scores.flatten(),
        "individual": individual_ids,
        "keypoint": keypoint_ids,
        "time": ds.coords["time"].values,
    }
    return napari_tracks, properties
