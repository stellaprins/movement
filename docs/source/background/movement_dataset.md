(target-dataset)=
# movement dataset

`movement` datasets are {class}`xarray.Dataset` objects, with some
added functionality for storing {term}`pose tracks` and related data.
Each dataset contains multiple data variables, dimensions, coordinates and
attributes.

![](../_static/dataset_structure.png)

## Dimensions and coordinates
The dataset has the following dimensions:
- `time`, with size equal to the number of frames in the video
- `individuals`, with size equal to the number of individuals in the video
- `keypoints`, with size equal to the number of tracked keypoints per individual
- `space`: the number of spatial dimensions, either 2 or 3

Appropriate coordinates, i.e. labels, are assigned to each dimension:
list of unique names (str) for `individuals` and `keypoints`,
`[x, y, (z)]` for `space`. The coordinates of the `time` dimension are
in seconds if `fps` is provided, otherwise they are in frame numbers.

## Data variables

Right after loading some predicted pose tracks into `movement`, the dataset
contains two data variables stored as {class}`xarray.DataArray` objects:
- `position`: with shape (`time`, `individuals`, `keypoints`, `space`)
- `confidence`: with shape (`time`, `individuals`, `keypoints`)

You can think of a `DataArray` as a {class}`numpy.ndarray` with `pandas`-style
indexing and labelling. We see that the `position` and `confidence` data
variables share 3 dimensions, and there are advantages to grouping them
together in the same `Dataset` object. To learn more about `xarray` data
structures, see the relevant
[documentation](xarray:user-guide/data-structures.html).

## Attributes

Attributes are a dictionary-like object that is used to store metadata
about the dataset. Right after loading some predicted pose tracks into
`movement`, the dataset contains the following attributes:
- `fps`: the number of frames per second in the video
- `time_unit`: the unit of the `time` coordinates, frames or seconds
- `source_software`: the software from which the pose tracks were loaded
- `source_file`: the file from which the pose tracks were loaded
