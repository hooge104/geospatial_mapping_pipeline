# Geospatial Mapping Pipeline
A fully automated geospatial pipeline. 

## How to get started
1. Sign up for a GEE account on https://signup.earthengine.google.com
2. Install Python 3 on your local machine (e.g. using Anaconda)
3. Install the GEE Python API using pip or conda. See details on https://developers.google.com/earth-engine/guides/python_install
4. Authenticate the GEE Python API
5. Install the following packages: `pandas`, `numpy`, `subprocess`, `time`, `datetime`, `scipy`, `sklearn`, `itertools`, `pathlib`, `math`, `matplotlib`, `tqdm` using your package manager
6. Recommended: when access to Google Cloud Storage is available: install gsutil

## Running the pipeline
The GMP is available as a executable python script, and can be run in the command line.

1. Create a folder on your local machine, to hold your project data
2. Place your data, as a `csv` file into this folder. This file should contain a column holding the response variable (of course...), and latitude/longitude. Default CRS: WGS84, EPSG:4326
3. Place the script into this folder and configure your project-specific details in the top part of the script.
4. Make sure to place the correct path of where `ee` and `gsutil` are installed on your local machine
5. Execute the script
