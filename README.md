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
Right now, you need to install the following python extensions

- gitpython 
- pandas
- requests
- folium    (This is needed if you want to create interactive maps)

If you installed Jupyter using the Anaconda (or miniconda) python distribution
then you can install all these using the following command on the command
line: `conda install gitpython pandas requests`

## Initialize Data collection by Running Collect_Data Notebook

The US Census Bureau population files, Apple and Google mobility data are
all single files and are retrieved on the fly.  However, The John Hopkins and
NY Times data on the spread of the disease consist of entire directory
structures full of files.  As such, I decided it made sense to pull their
entire directories off GitHub and use 'git pull' to keep it updated (since
that would be quick).

As of June 7, the `COVIDlib.collectors` library automatically handles creation
of local github repos as necessary.  **All you need to do to initialize the
data directories is to make sure to run the Collect_Data notebook or
corresponding python script before trying to generate vizualizations so all
the necessary COVID data is up to date.**