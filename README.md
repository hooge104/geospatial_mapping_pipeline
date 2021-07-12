# Geospatial Mapping Pipeline
A fully automated geospatial pipeline for ecologists.

## Citing this work
When using (part of) this modeling workflow, please cite Van den Hoogen et al., BioRxiv (2021). A geospatial mapping pipeline for ecologists, doi: [10.1101/2021.07.07.451145](https://www.biorxiv.org/content/10.1101/2021.07.07.451145v1)

## How to get started
1. Sign up for a GEE account on https://signup.earthengine.google.com
2. Install Python 3 on your local machine (e.g. using Anaconda)
3. Install the GEE Python API using pip or conda. See details on https://developers.google.com/earth-engine/guides/python_install
4. Authenticate the GEE Python API
5. Install the following packages: `pandas`, `numpy`, `subprocess`, `time`, `datetime`, `scipy`, `sklearn`, `itertools`, `pathlib`, `math`, `matplotlib` using your package manager
6. Recommended (when access to Google Cloud Storage is available): install `gsutil`

## Running the pipeline
The GMP is available as a executable python script, and can be run in the command line.

1. Create a folder on your local machine, to hold your project data
2. Place your data, as a csv file into this folder. This file should contain a column holding the response variable (of course...), and latitude+longitude. Default CRS: WGS84 EPSG:4326. At the moment, the pipeline assumes a continuous response variable, and will train a regression RF model.
3. Place the script into this folder and configure your project-specific details in the top part of the script (lines 19-82). Make sure to direct the bash functions (lines 54-55) to the correct path where `ee` and `gsutil` are installed on your local machine
5. Execute the script

## What to do next - how to visualise or export the results
When the script is ready, two GEE tasks started. When these have completed, you can either visualise them directly in the web GUI of GEE, or export them to Google Drive or Google Cloud Storage. Some code snippets (js) on how to visualise and export GEE assets below.

### Visualising a map on GEE
```
// Load image
var prediction = ee.ImageCollection('users/username/projectFolder/classProperty_Bootstrapped_MultibandImage').select('classProperty_PredictedEnsembleMean')

// Palette (see https://hooge104.github.io/color_palettes.html for more, or use the internet.)
var viridis = ["440154", "472D7B", "3B528B", "2C728E", "21908C", "27AD81", "5DC863", "AADC32", "FDE725"]

// Display map
// Make sure to adapt the min/max values to your image
Map.addLayer(prediction, {min:0,max:1000,palette:viridis}, 'Predicted image')
```

### Exporting an image from GEE to Google Drive
```
// Define export boundaries (this is a global geometry; modify to your needs, or draw a geometry using the shape/rectagnle tool on GEE)
var exportingGeometry = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], null, false);

// Export image to Google Drive
Export.image.toDrive({
    image: prediction,
    description: 'classProperty_PredictedEnsembleMean',
    crs: 'EPSG:4326', 
    crsTransform: [0.008333333333333333,0,-180,0,-0.008333333333333333,90],// 30 arc sec == 0.008333333333333333 degrees
    region: exportingGeometry,
    maxPixels: 1e13
});
```
An image is exported to Google Drive as a geotiff (.tif) by default. Use your favorite GIS software to visualise this file and create publication-ready figures. 

To export a FeatureCollection (as a csv), use `Export.table.toDrive`.

## Some notes in case Google Cloud Storage is not available
To automate the upload process, we routinely use GCS but as it's a paid service this might not be available to everyone. In case you want to bypass these steps, it's probably the easiest to manually upload the generated csv files (i.e., the training data file and the bootstrap samples) using the GEE web browser UI. You'll need to remove some parts of the script where `gsutil` is called. Just run the script until it complains that one of the assets can't be found, manually upload the file (make sure to give the GEE FeatureCollection the correct name and path) and re-run the script. You'll have to repeat this process a number of times; for the upload of the raw data and for the upload of the bootstrap subsamples. 

## Disclaimer
This script is still work in progress. If you spot any mistakes please let us know! We're also continously improving the workflow and developing new 'modules'. Watch this space for updates. 

We are aware that there are some steps that not necessarily follow GEE 'best practices'. Usually this is the result of a trade-off between computation time and flexibility. 
