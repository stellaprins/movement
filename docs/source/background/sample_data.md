(target-sample-data)=
# Sample data

`movement` includes some sample data files that you can use to
try out the package. These files contain predicted {term}`pose tracks` from
various [supported formats](target-supported-formats).
You can list the available sample data files using:

```python
from movement import sample_data

file_names = sample_data.list_sample_data()
print(file_names)
```

This will print a list of file names containing sample pose data.
Each file is prefixed with the name of the pose estimation software package
that was used to generate it - either "DLC", "SLEAP", or "LP".
To get the path to one of the sample files, you can use the
{func}`movement.sample_data.fetch_sample_data_path` function:

```python
file_path = sample_data.fetch_sample_data_path("DLC_two-mice.predictions.csv")
```
The first time you call this function, it will download the corresponding file
to your local machine and save it in the `~/.movement/data` directory. On
subsequent calls, it will simply return the path to that local file. You can
feed the path to the software-specific loading functions, as described
in the [supported formats](target-supported-formats) section.

Alternatively, you can skip the above step and load the data directly into
`movement`, by using the {func}`movement.sample_data.fetch_sample_data`
function:

```python
ds = sample_data.fetch_sample_data("DLC_two-mice.predictions.csv")
```

This will return a [movement dataset](target-dataset).
