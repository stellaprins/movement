"""Select data
==============

Select subsets of data from a movement dataset.

Use xarray's built-in indexing and selection features (See the
`xarray docs <https://docs.xarray.dev/en/stable/user-guide/indexing.html>`_
for more information) and convert the selections to numpy arrays or
pandas DataFrames.
"""

# %%
# Load a dataset
# --------------------------
# Here we load an example dataset that comes with the ``movement``.
# If you wish to load your own data from a file, refer to the
# :ref:`sphx_glr_auto_examples_load_pose_tracks.py` guide.
# sphinx_gallery_thumbnail_path = "_static/example_thumbnail.png"

from movement import sample_data

ds = sample_data.fetch_sample_data(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)

ds

# %%
# Select a specific data variable
# -------------------------------
# We can select a specific data variable from the dataset using the dot ``.``
# notation or square brackets ``[]``.
position = ds.position  # or ds["position"]
confidence = ds.confidence  # or ds["confidence"]

print(type(position), type(confidence))

# %%
# As we can see, each data variable is an :py:class:`xarray.DataArray` object,
# and we can also get its interactive preview by typing the variable name.
position

# %%
# Select data along each dimension
# --------------------------------
# We can select subsets of data along each dimension using the ``sel`` method.

# select a single individual
ds_single_ind = ds.sel(individuals="AEON3B_NTP")

# select a single keypoint
ds_single_kp = ds.sel(keypoints="centroid")  # here it's the only keypoint

# Select multiple individuals
ds_multiple_inds = ds.sel(individuals=["AEON3B_TP1", "AEON3B_TP2"])

# Select only the first 5 seconds of data
ds_first_5_sec = ds.sel(time=slice(0, 5))


# Combine multiple selections
ds_combined = ds.sel(
    individuals="AEON3B_NTP", keypoints="centroid", time=slice(5, 10)
)

ds_combined


# %%
# We can also select by index position using the ``isel`` method.

# Select the first 5 time points (frames)
ds_first_5_points = ds.isel(time=slice(0, 5))
print(f"n selected timepoints: {ds_first_5_points.time.size}")

# Select the second individual (they are 0-indexed)
ds_second_ind = ds.isel(individuals=1)
print(f"n selected individuals: {ds_second_ind.individuals.size}")


# %%
# All the above selections also work for the data variables.

position_x = position.sel(space="x")
position_x

# %%
# Convert to numpy array
# ----------------------
# We can convert the selected data variables to numpy arrays using the
# :py:attr:`xarray.DataArray.values` attribute.

# This is equivalent to position_x.data or np.array(position_x)
position_x_np = position_x.values
print(f"array type: {type(position_x_np)}")
print(f"array shape: {position_x_np.shape}")

# We can get rid of the redundant dimensions (with size 1) using squeeze
position_x_np_squeezed = position_x.squeeze().values
print(f"Squeezed array shape: {position_x_np_squeezed.shape}")

# .values also works for dimension coordinates
space_coords = position_x.space.values
print(f"Selected space coords: {space_coords}")

# %%
# Convert to pandas DataFrame
# ---------------------------
# We can convert the selected data variables to pandas DataFrames using the
# :py:meth:`xarray.DataArray.to_dataframe` method.

position_tp2_df = position_x.squeeze().to_dataframe(
    dim_order=["individuals", "time"]
)
position_tp2_df.head()


# %%
