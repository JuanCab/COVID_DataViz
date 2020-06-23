# COVID Visualization Project
# 
This is our CSIS 340 project.  The idea was to take public data sources on
COVID-19 and combine them into a new dashboard that could be potentially
useful for local planners.

All that is here right now is a single Jupyter notebook that accesses the
data. It exploits Pandas, a library for handling tabular data.  Pandas can be
really awesome in some ways and really annoying in others, so I am not wedded
to sticking with Pandas, we could just store the data in numpy arrays or
something simple like that.

## Requirements
Right now, you need to be running Python 3.7+ with the following python extensions for these notebooks to run.

- gitpython 
- pandas
- requests
- ipyleaflet (needed for widget based maps [replaces folium probably])
- ipywidgets (required to render widget controls in Jupyter)
- ipympl     (This is required to render matplotlib plots as widgets)

If you installed Jupyter using the Anaconda (or miniconda) python distribution
then you can install all these using the following command on the command
line: `conda install gitpython pandas requests ipyleaflet ipywidgets ipympl`

## Updating the Data Locally

Updates are probably most easily run from the command line by typing 

`python Collect_Data.py`  

(although the Jupyter Notebook `Collect_Data.ipynb` will accomplish the same).  This script pulls the entire US Census Bureau population files, Apple and Google mobility data on the fly. However, The John Hopkins data on the spread of the disease consist of entire directory structures full of files.  As such, it made sense to simply pull the GitHub repo to update the changes after the initial cloning.  The first time the script is run, it will create all the local directories necessary to store the data.

## Using The Dashboard

Once a day you should update the data using `Collect_Data.py`, after that, you can run the provided `Dashboard.ipynb` notebook to view the Dashboard locally.  There is also an `Examples.ipynb` to show a couple of the plotting and mapping commands in action in case you want to try to build your own custom dashboard.