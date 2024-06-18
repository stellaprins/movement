"""``attrs`` classes for validating file paths."""

import ast
import os
import re
from pathlib import Path
from typing import Literal

import h5py
import pandas as pd
from attrs import define, field, validators

from movement.utils.logging import log_error


@define
class ValidFile:
    """Class for validating file paths.

    Attributes
    ----------
    path : str or pathlib.Path
        Path to the file.
    expected_permission : {"r", "w", "rw"}
        Expected access permission(s) for the file. If "r", the file is
        expected to be readable. If "w", the file is expected to be writable.
        If "rw", the file is expected to be both readable and writable.
        Default: "r".
    expected_suffix : list of str
        Expected suffix(es) for the file. If an empty list (default), this
        check is skipped.

    Raises
    ------
    IsADirectoryError
        If the path points to a directory.
    PermissionError
        If the file does not have the expected access permission(s).
    FileNotFoundError
        If the file does not exist when `expected_permission` is "r" or "rw".
    FileExistsError
        If the file exists when `expected_permission` is "w".
    ValueError
        If the file does not have one of the expected suffix(es).

    """

    path: Path = field(converter=Path, validator=validators.instance_of(Path))
    expected_permission: Literal["r", "w", "rw"] = field(
        default="r", validator=validators.in_(["r", "w", "rw"]), kw_only=True
    )
    expected_suffix: list[str] = field(factory=list, kw_only=True)

    @path.validator
    def _path_is_not_dir(self, attribute, value):
        """Ensure that the path does not point to a directory."""
        if value.is_dir():
            raise log_error(
                IsADirectoryError,
                f"Expected a file path but got a directory: {value}.",
            )

    @path.validator
    def _file_exists_when_expected(self, attribute, value):
        """Ensure that the file exists (or not) as needed.

        This depends on the expected usage (read and/or write).
        """
        if "r" in self.expected_permission:
            if not value.exists():
                raise log_error(
                    FileNotFoundError, f"File {value} does not exist."
                )
        else:  # expected_permission is "w"
            if value.exists():
                raise log_error(
                    FileExistsError, f"File {value} already exists."
                )

    @path.validator
    def _file_has_access_permissions(self, attribute, value):
        """Ensure that the file has the expected access permission(s).

        Raises a PermissionError if not.
        """
        file_is_readable = os.access(value, os.R_OK)
        parent_is_writeable = os.access(value.parent, os.W_OK)
        if ("r" in self.expected_permission) and (not file_is_readable):
            raise log_error(
                PermissionError,
                f"Unable to read file: {value}. "
                "Make sure that you have read permissions.",
            )
        if ("w" in self.expected_permission) and (not parent_is_writeable):
            raise log_error(
                PermissionError,
                f"Unable to write to file: {value}. "
                "Make sure that you have write permissions.",
            )

    @path.validator
    def _file_has_expected_suffix(self, attribute, value):
        """Ensure that the file has one of the expected suffix(es)."""
        if self.expected_suffix and value.suffix not in self.expected_suffix:
            raise log_error(
                ValueError,
                f"Expected file with suffix(es) {self.expected_suffix} "
                f"but got suffix {value.suffix} instead.",
            )


@define
class ValidHDF5:
    """Class for validating HDF5 files.

    Attributes
    ----------
    path : pathlib.Path
        Path to the HDF5 file.
    expected_datasets : list of str or None
        List of names of the expected datasets in the HDF5 file. If an empty
        list (default), this check is skipped.

    Raises
    ------
    ValueError
        If the file is not in HDF5 format or if it does not contain the
        expected datasets.

    """

    path: Path = field(validator=validators.instance_of(Path))
    expected_datasets: list[str] = field(factory=list, kw_only=True)

    @path.validator
    def _file_is_h5(self, attribute, value):
        """Ensure that the file is indeed in HDF5 format."""
        try:
            with h5py.File(value, "r") as f:
                f.close()
        except Exception as e:
            raise log_error(
                ValueError,
                f"File {value} does not seem to be in valid" "HDF5 format.",
            ) from e

    @path.validator
    def _file_contains_expected_datasets(self, attribute, value):
        """Ensure that the HDF5 file contains the expected datasets."""
        if self.expected_datasets:
            with h5py.File(value, "r") as f:
                diff = set(self.expected_datasets).difference(set(f.keys()))
                if len(diff) > 0:
                    raise log_error(
                        ValueError,
                        f"Could not find the expected dataset(s) {diff} "
                        f"in file: {value}. ",
                    )


@define
class ValidDeepLabCutCSV:
    """Class for validating DeepLabCut-style .csv files.

    Attributes
    ----------
    path : pathlib.Path
        Path to the .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected DeepLabCut index column
        levels among its top rows.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def _csv_file_contains_expected_levels(self, attribute, value):
        """Ensure that the .csv file contains the expected index column levels.

        These are to be found among the top 4 rows of the file.
        """
        expected_levels = ["scorer", "bodyparts", "coords"]

        with open(value) as f:
            top4_row_starts = [f.readline().split(",")[0] for _ in range(4)]

            if top4_row_starts[3].isdigit():
                # if 4th row starts with a digit, assume single-animal DLC file
                expected_levels.append(top4_row_starts[3])
            else:
                # otherwise, assume multi-animal DLC file
                expected_levels.insert(1, "individuals")

            if top4_row_starts != expected_levels:
                raise log_error(
                    ValueError,
                    ".csv header rows do not match the known format for "
                    "DeepLabCut pose estimation output files.",
                )


@define
class ValidVIAtracksCSV:
    """Class for validating VIA tracks .csv files.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the VIA tracks .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not match the VIA tracks file requirements.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def csv_file_contains_expected_levels(self, attribute, value):
        """Ensure the VIA tracks .csv file contains the expected header."""
        expected_levels = [
            "filename",
            "file_size",
            "file_attributes",
            "region_count",
            "region_id",
            "region_shape_attributes",
            "region_attributes",
        ]

        with open(value) as f:
            header = f.readline().strip("\n").split(",")

            if header != expected_levels:
                raise log_error(
                    ValueError,
                    ".csv header row does not match the known format for "
                    "VIA tracks output files. "
                    f"Expected {expected_levels} but got {header}.",
                )

    @path.validator
    def csv_file_contains_valid_frame_numbers(self, attribute, value):
        """Ensure that the VIA tracks csv file contains valid frame numbers.

        This involves:
        - Checking that frame numbers are defined as a `file_attributes` or
          encoded in the image file `filename`.
        - Checking the frame number can be cast as an integer.
        - Checking that there are as many unique frame numbers as unique image
          files.

        If the frame number is included as part of the image file name, it is
        expected as an integer led by at least one zero, between "_" and ".",
        followed by the file extension.
        """
        df = pd.read_csv(value, sep=",", header=0)

        # Extract list of file attributes (dicts)
        file_attributes_dicts = [
            ast.literal_eval(d) for d in df.file_attributes
        ]

        # If 'frame' is a file_attribute for all frames:
        # extract frame number
        list_frame_numbers = []
        if all(["frame" in d for d in file_attributes_dicts]):
            for k_i, k in enumerate(file_attributes_dicts):
                try:
                    list_frame_numbers.append(int(k["frame"]))
                except Exception as e:
                    raise log_error(
                        ValueError,
                        f"{df.filename.iloc[k_i]} (row {k_i}): "
                        "'frame' file attribute cannot be cast as an integer. "
                        f"Please review the file attributes: {k}.",
                    ) from e

        # else: extract frame number from filename.
        else:
            pattern = r"_(0\d*)\.\w+$"

            for f_i, f in enumerate(df["filename"]):
                regex_match = re.search(pattern, f)
                if regex_match:  # if there is a pattern match
                    list_frame_numbers.append(
                        int(regex_match.group(1))  # type: ignore
                        # the match will always be castable as integer
                    )
                else:
                    raise log_error(
                        ValueError,
                        f"{f} (row {f_i}): "
                        "a frame number could not be extracted from the "
                        "filename. If included in the filename, the frame "
                        "number is expected as a zero-padded integer between "
                        "an underscore '_' and the file extension "
                        "(e.g. img_00234.png).",
                    )

        # Check we have as many unique frame numbers as unique image files
        if len(set(list_frame_numbers)) != len(df.filename.unique()):
            raise log_error(
                ValueError,
                "The number of unique frame numbers does not match the number "
                "of unique image files. Please review the VIA tracks csv file "
                "and ensure a unique frame number is defined for each file. ",
            )

    @path.validator
    def csv_file_contains_tracked_bboxes(self, attribute, value):
        """Ensure that the VIA tracks csv contains tracked bounding boxes.

        This involves:
        - Checking that the bounding boxes are defined as rectangles.
        - Checking that the bounding boxes have all geometric parameters
          (["x", "y", "width", "height"]).
        - Checking that the bounding boxes have a track ID defined.
        - Checking that the track ID can be cast as an integer.
        """
        df = pd.read_csv(value, sep=",", header=0)

        for row in df.itertuples():
            row_region_shape_attrs = ast.literal_eval(
                row.region_shape_attributes
            )
            row_region_attrs = ast.literal_eval(row.region_attributes)

            # check annotation is a rectangle
            if row_region_shape_attrs["name"] != "rect":
                raise log_error(
                    ValueError,
                    f"{row.filename} (row {row.Index}): "
                    "bounding box shape must be 'rect' (rectangular) "
                    "but instead got "
                    f"'{row_region_shape_attrs['name']}'.",
                )

            # check all geometric parameters for the box are defined
            if not all(
                [
                    key in row_region_shape_attrs
                    for key in ["x", "y", "width", "height"]
                ]
            ):
                raise log_error(
                    ValueError,
                    f"{row.filename} (row {row.Index}): "
                    f"at least one bounding box shape parameter is missing. "
                    "Expected 'x', 'y', 'width', 'height' to exist as "
                    "'region_shape_attributes', but got "
                    f"'{list(row_region_shape_attrs.keys())}'.",
                )

            # check track ID is defined
            if "track" not in row_region_attrs:
                raise log_error(
                    ValueError,
                    f"{row.filename} (row {row.Index}): "
                    "bounding box does not have a 'track' attribute defined "
                    "under 'region_attributes'. "
                    "Please review the VIA tracks csv file.",
                )

            # check track ID is castable as an integer
            try:
                int(row_region_attrs["track"])
            except Exception as e:
                raise log_error(
                    ValueError,
                    f"{row.filename} (row {row.Index}): "
                    "the track ID for the bounding box cannot be cast "
                    "as an integer. Please review the VIA tracks csv file.",
                ) from e

    @path.validator
    def csv_file_contains_unique_track_IDs_per_filename(
        self, attribute, value
    ):
        """Ensure the VIA tracks csv contains unique track IDs per filename.

        It check bounding boxes IDs are only defined once per image file.
        """
        df = pd.read_csv(value, sep=",", header=0)

        list_unique_filenames = list(set(df.filename))
        for file in list_unique_filenames:
            df_one_filename = df.loc[df["filename"] == file]

            list_track_IDs_one_filename = [
                int(ast.literal_eval(row.region_attributes)["track"])
                for row in df_one_filename.itertuples()
            ]

            if len(set(list_track_IDs_one_filename)) != len(
                list_track_IDs_one_filename
            ):
                raise log_error(
                    ValueError,
                    f"{file}: "
                    "multiple bounding boxes in this file "
                    "have the same track ID. "
                    "Please review the VIA tracks csv file.",
                )
