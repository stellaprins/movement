"""Convert Pose Tracks across formats
=====================================

Split a multi-animal SLEAP file into single-animal DeepLabCut files.

This example demonstrates how to load pose tracks predicted by one pose
tracking framework (SLEAP) and convert them to another format (DeepLabCut).
During, this process, we split the multi-animal input file into single-animal
output files. This could be useful when you want to use some downstream
analysis tool (e.g. a behaviour classifier) that only supports single-animal
data.
"""

# %%
# Load a multi-animal file from SLEAP
# -----------------------------------
# Here we load an example dataset that comes with the ``movement``.
# If you wish to load your own data from a file, replace the file path,
# e.g., :code:`file_path = "/path/to/my/data.h5"`.
# sphinx_gallery_thumbnail_path = "_static/example_thumbnail.png"

from movement import sample_data
from movement.io import load_poses

file_path = sample_data.fetch_sample_data_path(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)
ds = load_poses.from_sleap_file(file_path, fps=50)

print(f"Inidividuals: {ds.individuals.values}")

# We see that the dataset contains data for three individuals (``AEON3B_NTP``,
# ``AEON3B_TP1`` and ``AEON3B_TP2``).

# %%
# Save single-animal files in DeepLabCut format
# ---------------------------------------------
# Here we'll use the :py:func:`movement.io.save_poses.to_dlc_file` function
# with ``split_individuals=True``.

import tempfile

from movement.io import save_poses

# Create a temporary directory to save the files
# replace this with your desired directory
save_dir = tempfile.mkdtemp()

save_poses.to_dlc_file(
    ds,
    file_path=f"{save_dir}/single_animal_dlc.h5",
    split_individuals=True,
)

# %%
# This will produce 3 files with the individual names as suffixes (before the
# file extension).

single_animal_files = [
    "single_animal_dlc_AEON3B_NTP.h5",
    "single_animal_dlc_AEON3B_TP1.h5",
    "single_animal_dlc_AEON3B_TP2.h5",
]
single_animal_file_paths = [f"{save_dir}/{f}" for f in single_animal_files]

# %%
# Verify the saved files
# ----------------------
# Let's load the saved files to verify the split was successful.

for file in single_animal_file_paths:
    ds = load_poses.from_dlc_file(file)
    print(f"Loaded file: {file}")
    print(f"Individuals: {ds.individuals.values}")

# display the last-loaded dataset
ds


# %%
